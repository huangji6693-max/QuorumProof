#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use soroban_sdk::{testutils::Address as _, Address, Bytes, Env};
use zk_verifier::groth16_test_prover::{self, ToyVerifyingKey};
use zk_verifier::{ClaimType, ZkVerifierContract, ZkVerifierContractClient, GROTH16_BLS_PROOF_LEN};

/// Fuzz input covering verify_claim with arbitrary proof bytes and claim types.
#[derive(Arbitrary, Debug)]
struct FuzzInput {
    proof: Vec<u8>,
    claim_type_idx: u8,
    credential_id: u64,
    /// Selects the verification mode the oracle is checked under:
    /// `false` = no verifying key registered (fail-closed path),
    /// `true`  = a real Groth16 verifying key is registered.
    real_mode: bool,
    /// Seed for the toy verifying key and its valid proof (real mode only).
    vk_seed: u64,
}

/// Public-input count the contract's `verify_claim` encodes: credential_id and claim discriminant.
const CLAIM_PUBLIC_INPUTS: u32 = 2;

fn claim_discriminant(claim_type: &ClaimType) -> u64 {
    match claim_type {
        ClaimType::HasDegree => 1,
        ClaimType::HasLicense => 2,
        ClaimType::HasEmploymentHistory => 3,
        ClaimType::HasCertification => 4,
        ClaimType::HasResearchPublication => 5,
    }
}

/// Known-invalid proofs the real verifier must reject for `(credential_id, claim_type)`.
/// Each entry is a distinct failure class: malformed length, degenerate points,
/// a corrupted valid proof, and a valid proof for a different statement.
fn known_invalid_proofs(
    env: &Env,
    toy: &ToyVerifyingKey,
    toy_seed: u64,
    credential_id: u64,
    claim: u64,
    valid: &[u8],
) -> Vec<(&'static str, Bytes)> {
    let proof_len = GROTH16_BLS_PROOF_LEN as usize;
    let mut flip_first = valid.to_vec();
    flip_first[0] ^= 0x01;
    let mut flip_last = valid.to_vec();
    *flip_last.last_mut().unwrap() ^= 0x01;
    let mut long = valid.to_vec();
    long.push(0);

    let other_cid = groth16_test_prover::generate_proof(
        env,
        toy,
        toy_seed.wrapping_add(1),
        &[credential_id.wrapping_add(1), claim],
    );
    let other_claim_value = (claim % 5) + 1;
    let other_claim = groth16_test_prover::generate_proof(
        env,
        toy,
        toy_seed.wrapping_add(2),
        &[credential_id, other_claim_value],
    );

    vec![
        ("empty", Bytes::new(env)),
        ("one_byte", Bytes::from_slice(env, &[0u8])),
        ("truncated", Bytes::from_slice(env, &valid[..proof_len - 1])),
        ("oversized", Bytes::from_slice(env, &long)),
        ("all_zero", Bytes::from_slice(env, &vec![0u8; proof_len])),
        ("all_ff", Bytes::from_slice(env, &vec![0xffu8; proof_len])),
        ("corrupt_leading_point", Bytes::from_slice(env, &flip_first)),
        ("corrupt_trailing_point", Bytes::from_slice(env, &flip_last)),
        ("wrong_credential_id", other_cid.proof),
        ("wrong_claim_type", other_claim.proof),
    ]
}

fuzz_target!(|input: FuzzInput| {
    let env = Env::default();
    env.mock_all_auths();

    let contract_id = env.register_contract(None, ZkVerifierContract);
    let client = ZkVerifierContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    client.initialize(&admin);

    let qp_id = Address::generate(&env);
    let proof = Bytes::from_slice(&env, &input.proof);

    let claim_type = match input.claim_type_idx % 5 {
        0 => ClaimType::HasDegree,
        1 => ClaimType::HasLicense,
        2 => ClaimType::HasEmploymentHistory,
        3 => ClaimType::HasCertification,
        _ => ClaimType::HasResearchPublication,
    };

    // Returns `None` if the contract panicked on this input. Fuzz inputs may
    // legitimately trigger contract panics, so those are skipped below; the
    // known-invalid corpus treats a panic as a failure instead.
    let verify = |p: &Bytes| -> Option<bool> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            client.verify_claim(&admin, &qp_id, &input.credential_id, &claim_type, p)
        }))
        .ok()
    };

    if !input.real_mode {
        // Mode 1: no verifying key registered. The verifier must fail closed
        // for every proof, including non-empty ones.
        if let Some(result) = verify(&proof) {
            assert!(!result, "verify_claim must fail closed when no Groth16 key is registered");
        }
        return;
    }

    // Mode 2: real Groth16 verification with a registered key.
    let toy = groth16_test_prover::generate_vk(&env, input.vk_seed, CLAIM_PUBLIC_INPUTS);
    client.set_groth16_verifying_key(&admin, &toy.vk_hash, &toy.vk);

    let claim = claim_discriminant(&claim_type);
    let valid = groth16_test_prover::generate_proof(&env, &toy, input.vk_seed, &[input.credential_id, claim]);

    // Sanity: a genuinely valid proof for this statement must be accepted,
    // otherwise the rejection checks below would pass for the wrong reason.
    assert_eq!(
        verify(&valid.proof),
        Some(true),
        "a valid Groth16 proof for this statement must be accepted"
    );

    // Known-invalid corpus: every entry must be rejected without panicking.
    let valid_bytes = valid.proof.to_alloc_vec();
    for (name, bad) in known_invalid_proofs(
        &env,
        &toy,
        input.vk_seed,
        input.credential_id,
        claim,
        &valid_bytes,
    ) {
        assert_eq!(verify(&bad), Some(false), "known-invalid proof `{name}` must be rejected");
    }

    // Arbitrary fuzz input: anything other than the valid proof must be rejected.
    if let Some(result) = verify(&proof) {
        if input.proof != valid_bytes {
            assert!(!result, "arbitrary non-valid proof must not pass real verification");
        }
    }
});
