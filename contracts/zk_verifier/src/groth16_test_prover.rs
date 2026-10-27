//! Test-only Groth16 fixture generator.
//!
//! Rather than compiling a real circuit and running a trusted-setup
//! ceremony, this builds a genuinely algebraically-satisfying `(vk, ic,
//! proof, public_inputs)` instance directly: the "toxic waste" scalars
//! (`alpha`, `beta`, `gamma`, `delta`, the `ic` coefficients) and the
//! proof's `A`/`B` exponents are picked freely, then `C` is solved for so
//! the verification equation holds exactly. This is the same technique a
//! simulator uses in the Groth16 zero-knowledge proof, and is the standard
//! way to unit-test a pairing verifier's algebra independent of any
//! specific circuit — see [`crate::groth16`] for the equation being
//! satisfied, and `plonk_test_prover` for the analogous (full toy-circuit)
//! approach taken for PLONK.
#![cfg(any(test, feature = "testutils"))]
extern crate std;

use std::vec::Vec;

use bls12_381::{G1Affine, G2Affine, Scalar};

use crate::groth16::{G1_LEN, G2_LEN, PROOF_LEN, SCALAR_LEN};
use crate::Groth16VerifyingKey;
use soroban_sdk::{Bytes, BytesN, Env, Vec as SVec};

/// A registered toy verifying key plus the "circuit" (IC) scalars needed to
/// generate further proofs against it.
pub struct ToyVerifyingKey {
    pub vk: Groth16VerifyingKey,
    pub vk_hash: BytesN<32>,
    alpha_s: Scalar,
    gamma_s: Scalar,
    delta_s: Scalar,
    ic_scalars: Vec<Scalar>,
    beta_s: Scalar,
}

/// A generated toy proof plus its matching public inputs.
pub struct ToyProof {
    pub public_inputs: Bytes,
    pub proof: Bytes,
}

/// A fixed, deterministic (non-random) toxic-waste scalar for the toy
/// trusted setup. Fine for tests — a real deployment must run an actual
/// multi-party ceremony.
fn toxic_waste(seed: u64) -> Scalar {
    let raw = 0x51e_c0de_u64.wrapping_mul(seed).wrapping_add(424_242);
    Scalar::from(if raw == 0 { 1 } else { raw })
}

/// Generates a toy Groth16 verifying key with `num_public_inputs`
/// public-input wires, derived deterministically from `seed`. Multiple
/// proofs generated via [`generate_proof`] against the same [`ToyVerifyingKey`]
/// all share one `vk_hash` — the realistic "many credentials, one circuit"
/// shape needed to exercise batch/aggregate verification.
pub fn generate_vk(env: &Env, seed: u64, num_public_inputs: u32) -> ToyVerifyingKey {
    let alpha_s = toxic_waste(seed.wrapping_add(1));
    let beta_s = toxic_waste(seed.wrapping_add(2));
    let gamma_s = toxic_waste(seed.wrapping_add(3));
    let delta_s = toxic_waste(seed.wrapping_add(4));

    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();

    let alpha_g1 = G1Affine::from(g1 * alpha_s);
    let beta_g2 = G2Affine::from(g2 * beta_s);
    let gamma_g2 = G2Affine::from(g2 * gamma_s);
    let delta_g2 = G2Affine::from(g2 * delta_s);

    let mut ic_scalars: Vec<Scalar> = Vec::new();
    let mut ic_sdk: SVec<BytesN<48>> = SVec::new(env);
    for i in 0..=num_public_inputs {
        let ic_s = toxic_waste(seed.wrapping_add(100 + i as u64));
        ic_scalars.push(ic_s);
        let ic_p = G1Affine::from(g1 * ic_s);
        ic_sdk.push_back(BytesN::from_array(env, &ic_p.to_compressed()));
    }

    let vk = Groth16VerifyingKey {
        alpha_g1: BytesN::from_array(env, &alpha_g1.to_compressed()),
        beta_g2: BytesN::from_array(env, &beta_g2.to_compressed()),
        gamma_g2: BytesN::from_array(env, &gamma_g2.to_compressed()),
        delta_g2: BytesN::from_array(env, &delta_g2.to_compressed()),
        ic: ic_sdk,
    };

    // Reuse the contract's own hashing logic so this fixture's `vk_hash`
    // always matches what `set_groth16_verifying_key` recomputes and checks.
    let vk_hash = crate::ZkVerifierContract::groth16_vk_hash(env.clone(), vk.clone());

    ToyVerifyingKey {
        vk,
        vk_hash,
        alpha_s,
        gamma_s,
        delta_s,
        ic_scalars,
        beta_s,
    }
}

/// Generates a genuinely valid Groth16 proof against `vk` for the given
/// `public_input_values`, using `proof_seed` for the proof's own (A, B)
/// exponents — distinct `proof_seed`s against the same `vk` yield distinct
/// but equally valid proofs (simulating different holders' credentials
/// under one circuit).
pub fn generate_proof(env: &Env, vk: &ToyVerifyingKey, proof_seed: u64, public_input_values: &[u64]) -> ToyProof {
    assert_eq!(
        public_input_values.len() + 1,
        vk.ic_scalars.len(),
        "public_input_values length must match vk's public-input count"
    );

    let mut a_s = toxic_waste(proof_seed.wrapping_add(5));
    let b_s = toxic_waste(proof_seed.wrapping_add(6));

    let g1 = G1Affine::generator();
    let g2 = G2Affine::generator();

    let public_input_scalars: Vec<Scalar> = public_input_values.iter().map(|v| Scalar::from(*v)).collect();

    let mut vk_x_s = vk.ic_scalars[0];
    for (i, pi) in public_input_scalars.iter().enumerate() {
        vk_x_s += *pi * vk.ic_scalars[i + 1];
    }

    let delta_inv = Option::<Scalar>::from(vk.delta_s.invert()).expect("delta must be invertible");
    // Solve for C such that a*b == alpha*beta + vk_x*gamma + c_exp*delta.
    let mut num = a_s * b_s - vk.alpha_s * vk.beta_s - vk_x_s * vk.gamma_s;
    if num == Scalar::zero() {
        a_s += Scalar::one();
        num = a_s * b_s - vk.alpha_s * vk.beta_s - vk_x_s * vk.gamma_s;
    }
    let a = G1Affine::from(g1 * a_s);
    let b = G2Affine::from(g2 * b_s);
    let c_exp = num * delta_inv;
    let c = G1Affine::from(g1 * c_exp);

    let mut proof_buf = [0u8; PROOF_LEN as usize];
    proof_buf[0..G1_LEN].copy_from_slice(&a.to_compressed());
    proof_buf[G1_LEN..G1_LEN + G2_LEN].copy_from_slice(&b.to_compressed());
    proof_buf[G1_LEN + G2_LEN..PROOF_LEN as usize].copy_from_slice(&c.to_compressed());

    let mut pi_bytes = std::vec![0u8; public_input_scalars.len() * SCALAR_LEN];
    for (i, pi) in public_input_scalars.iter().enumerate() {
        pi_bytes[i * SCALAR_LEN..(i + 1) * SCALAR_LEN].copy_from_slice(&pi.to_bytes());
    }

    ToyProof {
        public_inputs: Bytes::from_slice(env, &pi_bytes),
        proof: Bytes::from_slice(env, &proof_buf),
    }
}

/// A generated toy proof plus every piece of on-chain state needed to
/// register and verify it — convenience wrapper around
/// [`generate_vk`] + [`generate_proof`] for the common single-proof case.
pub struct ToyProofFixture {
    pub vk: Groth16VerifyingKey,
    pub vk_hash: BytesN<32>,
    pub public_inputs: Bytes,
    pub proof: Bytes,
}

/// Generates a genuinely valid Groth16 proof for a fresh toy verifying key
/// with `num_public_inputs` public-input wires (assigned values `1, 2, ..,
/// num_public_inputs`), derived from `seed`.
pub fn generate_valid_proof(env: &Env, seed: u64, num_public_inputs: u32) -> ToyProofFixture {
    let vk = generate_vk(env, seed, num_public_inputs);
    let values: Vec<u64> = (1..=num_public_inputs as u64).collect();
    let proof = generate_proof(env, &vk, seed, &values);
    ToyProofFixture {
        vk: vk.vk,
        vk_hash: vk.vk_hash,
        public_inputs: proof.public_inputs,
        proof: proof.proof,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClaimType, ZkVerifierContract, ZkVerifierContractClient};
    use soroban_sdk::{testutils::Address as _, Address};

    #[test]
    fn test_toxic_waste_never_produces_zero_scalar_on_wrapping_zero_seed() {
        // Solve 0x51ec0de * seed + 424242 == 0 (mod 2^64) via Newton's method on 0x28f606f
        let odd_factor: u64 = 0x51e_c0de_u64 >> 1;
        let mut inv: u64 = 1;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(odd_factor.wrapping_mul(inv)));
        }
        let zero_producing_seed = inv.wrapping_mul((0u64.wrapping_sub(424_242)) >> 1);
        assert_eq!(
            0x51e_c0de_u64
                .wrapping_mul(zero_producing_seed)
                .wrapping_add(424_242),
            0
        );

        // Even when raw wrapping arithmetic hits 0, toxic_waste and generate_vk must succeed
        let s = toxic_waste(zero_producing_seed);
        assert_ne!(s, Scalar::zero());

        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);
        client.initialize(&admin);

        let vk_seed = zero_producing_seed.wrapping_sub(4); // hits delta_s = toxic_waste(seed + 4)
        let toy = generate_vk(&env, vk_seed, 2);
        client.set_groth16_verifying_key(&admin, &toy.vk_hash, &toy.vk);

        let qp_id = Address::generate(&env);
        let valid = generate_proof(&env, &toy, vk_seed, &[0, 1]);
        assert!(client.verify_claim(&admin, &qp_id, &0, &ClaimType::HasDegree, &valid.proof));
    }
}

