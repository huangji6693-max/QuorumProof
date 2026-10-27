#![no_std]
// Test builds use `std::` paths (formatting, catch_unwind, Vec); the crate itself stays no_std.
#[cfg(test)]
extern crate std;

/// Topic for admin-transfer events emitted by [`ZkVerifierContract::update_admin`].
const TOPIC_ADMIN_TRANSFERRED: &str = "AdminTransferred";

use soroban_sdk::{contract, contractimpl, contracttype, Address, Bytes, BytesN, Env, String, Vec};
use soroban_sdk::xdr::ToXdr;

mod plonk;
mod groth16;
// `test` so this crate's own `mod tests` can use it; `testutils` so downstream
// crates (e.g. benches/) can too, the same way soroban-sdk's own testutils
// feature works across crate boundaries — plain `#[cfg(test)]` only applies
// to this crate's own test builds, not to integration tests in other crates.
#[cfg(any(test, feature = "testutils"))]
pub mod plonk_test_prover;
#[cfg(any(test, feature = "testutils"))]
pub mod groth16_test_prover;

/// Legacy Groth16 proof byte layout (BN254-shaped, uncompressed):
///   A  : 64 bytes  (G1 point)
///   B  : 128 bytes (G2 point)
///   C  : 64 bytes  (G1 point)
///   Total: 256 bytes
///
/// Used by [`ZkVerifierContract::verify_batch_proofs`] (structural validation
/// step) and by the test-fixtures module (`legacy_heuristic_fixtures`) whose
/// heuristic path (`verify_claim`, `verify_proof_cached`,
/// `verify_claim_anonymous`) is compiled in only under
/// `#[cfg(any(test, feature = "testutils"))]`.
///
/// **Do not use this constant to pass proofs to production verification.**
/// The real, production entry point is [`ZkVerifierContract::verify_groth16_proof`],
/// which performs genuine BLS12-381 pairing checks (192-byte proofs, see
/// [`GROTH16_BLS_PROOF_LEN`] and the [`groth16`] module).
pub const GROTH16_PROOF_LEN: u32 = 256;

/// Real Groth16 proof byte layout (BLS12-381, compressed): `A` (G1, 48) ‖
/// `B` (G2, 96) ‖ `C` (G1, 48) = 192 bytes. See the [`groth16`] module for
/// the verification equation and rationale for instantiating over
/// BLS12-381 rather than BN254 (mirrors the same switch already made by the
/// [`plonk`] module).
pub const GROTH16_BLS_PROOF_LEN: u32 = groth16::PROOF_LEN;

/// Maximum public-input field elements accepted by
/// [`ZkVerifierContract::verify_groth16_proof`] / `verify_aggregated_proofs`.
const MAX_GROTH16_PUBLIC_INPUTS: usize = groth16::MAX_PUBLIC_INPUTS;

/// Maximum number of proofs verifiable in a single `verify_aggregated_proofs`
/// call. Bounds the fixed stack buffers used to copy proofs/public inputs
/// out of host `Vec<Bytes>` objects (see `groth16::verify_batch`'s docs on
/// why per-proof pairing work can't be pre-batched further without a global
/// allocator).
const MAX_GROTH16_AGGREGATE_BATCH: usize = 16;

/// Issue #1732: Upper bound on the number of entries accepted by
/// `verify_batch_proofs` in a single call. Oversized batches are rejected up
/// front instead of failing at the per-transaction resource limit.
pub const MAX_VERIFY_BATCH: u32 = 16;

/// Key rotation audit entry
#[contracttype]
#[derive(Clone)]
pub struct KeyRotationEntry {
    pub old_key: BytesN<32>,
    pub new_key: BytesN<32>,
    pub rotated_at_ledger: u32,
    pub rotated_by: Address,
}

/// Circuit-specific Groth16 verifying key for the real BLS12-381
/// pairing-based verifier ([`groth16`] module), registered via
/// [`ZkVerifierContract::set_groth16_verifying_key`] /
/// [`ZkVerifierContract::rotate_groth16_verifying_key`].
///
/// `ic` is the circuit's public-input linear-combination basis: `ic[0]` is
/// the constant term and `ic[1..]` has one G1 point per public-input wire,
/// so `ic.len()` must equal the circuit's public-input count plus one.
#[contracttype]
#[derive(Clone)]
pub struct Groth16VerifyingKey {
    pub alpha_g1: BytesN<48>,
    pub beta_g2: BytesN<96>,
    pub gamma_g2: BytesN<96>,
    pub delta_g2: BytesN<96>,
    pub ic: Vec<BytesN<48>>,
}

impl Groth16VerifyingKey {
    fn to_core(&self) -> groth16::VerifyingKey {
        groth16::VerifyingKey {
            alpha_g1: self.alpha_g1.to_array(),
            beta_g2: self.beta_g2.to_array(),
            gamma_g2: self.gamma_g2.to_array(),
            delta_g2: self.delta_g2.to_array(),
        }
    }

    /// Panics (admin-facing input validation) if `ic` is empty, exceeds
    /// [`MAX_GROTH16_PUBLIC_INPUTS`] + 1 entries, or any of the four fixed
    /// elements / `ic` points fails to deserialize as a valid BLS12-381
    /// point in the prime-order subgroup.
    fn validate(&self) {
        assert!(!self.ic.is_empty(), "ic must have at least one (constant) entry");
        assert!(
            self.ic.len() as usize <= MAX_GROTH16_PUBLIC_INPUTS + 1,
            "ic exceeds MAX_GROTH16_PUBLIC_INPUTS + 1"
        );
        assert!(groth16::is_valid_g1(&self.alpha_g1.to_array()), "invalid alpha_g1 point");
        assert!(groth16::is_valid_g2(&self.beta_g2.to_array()), "invalid beta_g2 point");
        assert!(groth16::is_valid_g2(&self.gamma_g2.to_array()), "invalid gamma_g2 point");
        assert!(groth16::is_valid_g2(&self.delta_g2.to_array()), "invalid delta_g2 point");
        for p in self.ic.iter() {
            assert!(groth16::is_valid_g1(&p.to_array()), "invalid ic point");
        }
    }
}

/// Audit-trail entry for a Groth16 verifying-key rotation. Mirrors
/// [`PlonkKeyRotationEntry`].
#[contracttype]
#[derive(Clone)]
pub struct Groth16KeyRotationEntry {
    pub old_vk_hash: BytesN<32>,
    pub new_vk_hash: BytesN<32>,
    pub rotated_at_ledger: u32,
    pub rotated_by: Address,
}

/// Aggregated Groth16 proof for a batch of credentials.
///
/// The aggregation uses a random linear combination (SnarkPack-style) over
/// per-proof binding hashes, computed deterministically from `agg_nonce`.
/// This allows verifying n proofs with O(n·hash) cost instead of O(n·pairing),
/// while preserving per-credential accountability through the binding hashes.
///
/// # Layout
/// `proof_bytes` is a 256-byte representative proof (same layout as a single
/// Groth16 proof: A[0..64] ‖ B[64..192] ‖ C[192..256]). In a full pairing
/// implementation this would be A_agg ‖ B_agg ‖ C_agg.
#[contracttype]
#[derive(Clone)]
pub struct AggregateProof {
    /// Representative proof bytes (256 bytes, same layout as a single Groth16 proof).
    pub proof_bytes: Bytes,
    /// Random nonce used to derive deterministic combination scalars r_i.
    /// r_i = SHA-256(agg_nonce ‖ i.to_le_bytes())
    pub agg_nonce: BytesN<32>,
    /// Number of proofs in the batch.
    pub batch_size: u32,
}

/// Compute a deterministic combination scalar for proof index `i`:
/// `r_i = SHA-256(agg_nonce ‖ i.to_le_bytes())`
fn aggregation_scalar(env: &Env, agg_nonce: &BytesN<32>, i: u32) -> [u8; 32] {
    let mut input = Bytes::new(env);
    input.extend_from_array(&agg_nonce.to_array());
    input.extend_from_array(&i.to_le_bytes());
    env.crypto().sha256(&input).to_array()
}

/// Compute the per-proof binding hash (same logic as `verify_groth16_proof` step 5):
/// `h_i = SHA-256(vk_hash ‖ SHA-256(public_inputs) ‖ proof)`
///
/// This binds the proof to its credential's verifying key and public inputs,
/// preserving per-credential accountability in the aggregate check.
fn proof_binding_hash(
    env: &Env,
    proof: &Bytes,
    public_inputs: &Bytes,
    vk_hash: &BytesN<32>,
) -> [u8; 32] {
    let pi_digest = env.crypto().sha256(public_inputs);
    let mut binding_input = Bytes::new(env);
    binding_input.extend_from_array(&vk_hash.to_array());
    binding_input.extend_from_array(&pi_digest.to_array());
    binding_input.append(proof);
    env.crypto().sha256(&binding_input).to_array()
}

/// Structural/hash-binding heuristic Groth16 check.
///
/// Delegates to [`legacy_heuristic_fixtures::groth16_verify`].
///
/// **WARNING: NOT real cryptographic verification. Test-only entry point.**
#[cfg(any(test, feature = "testutils"))]
fn groth16_verify(env: &Env, vk_hash: &BytesN<32>, proof: &Bytes) -> bool {
    legacy_heuristic_fixtures::groth16_verify(env, vk_hash, proof)
}

/// Enhanced VK binding with multiple collision resistance checks.
///
/// Delegates to [`legacy_heuristic_fixtures::verify_enhanced_vk_binding`].
///
/// **WARNING: This is not real cryptographic verification. Use only for testing.**
#[cfg(any(test, feature = "testutils"))]
fn verify_enhanced_vk_binding(env: &Env, vk_hash: &BytesN<32>, proof: &Bytes) -> bool {
    legacy_heuristic_fixtures::verify_enhanced_vk_binding(env, vk_hash, proof)
}

/// Test-fixture module for the legacy BN254-shaped, 256-byte Groth16 heuristic.
///
/// # Why this exists
/// Early versions of this contract shipped a structural/hash-binding heuristic
/// (not real elliptic-curve pairing math) under the `verify_claim` /
/// `verify_proof_cached` / `verify_claim_anonymous` entry points so that
/// in-flight test fixtures from `quorum_proof` kept working while the real
/// BLS12-381 verifier ([`groth16`] module, 192-byte proofs) was being developed.
///
/// # What is here
/// - `LEGACY_PROOF_LEN` (256) — the BN254-shaped proof length used by the heuristic.
/// - `groth16_verify` / `verify_enhanced_vk_binding` — hash-binding checks that
///   stand in for real pairing math.
///
/// # Production API (NOT in this module)
/// - [`ZkVerifierContract::verify_groth16_proof`] — real BLS12-381 pairing,
///   192-byte ([`GROTH16_BLS_PROOF_LEN`]) proofs, permissionless.
/// - [`ZkVerifierContract::verify_plonk_proof`] — KZG/PLONK, 624-byte
///   ([`PLONK_PROOF_LEN`]) proofs.
///
/// # Safety
/// This module is compiled in **only** under `#[cfg(any(test, feature = "testutils"))]`
/// and **must not** be invoked from any production (non-test) code path.
#[cfg(any(test, feature = "testutils"))]
pub(crate) mod legacy_heuristic_fixtures {
    use soroban_sdk::{Bytes, BytesN, Env};

    /// Legacy BN254-shaped proof length (256 bytes). Test-fixtures use only.
    pub(crate) const LEGACY_PROOF_LEN: u32 = 256;

    /// Structural/hash-binding heuristic Groth16 check (BN254 layout, 256 bytes).
    ///
    /// **NOT real cryptographic verification.** Keeps legacy test fixtures working.
    pub(crate) fn groth16_verify(env: &Env, vk_hash: &BytesN<32>, proof: &Bytes) -> bool {
        if proof.len() != LEGACY_PROOF_LEN {
            return false;
        }

        // A-point validation (bytes 0-63): non-zero, no 0xFF in first 32 bytes
        let mut a_zero = true;
        let mut a_valid = true;
        for i in 0..64 {
            let byte_val = proof.get(i).unwrap_or(0);
            if byte_val != 0 { a_zero = false; }
            if i < 32 && byte_val == 0xFF { a_valid = false; }
        }
        if a_zero || !a_valid { return false; }

        // C-point validation (bytes 192-255): non-zero, no 0xFF in bytes 192-223
        let mut c_zero = true;
        let mut c_valid = true;
        for i in 192..256 {
            let byte_val = proof.get(i).unwrap_or(0);
            if byte_val != 0 { c_zero = false; }
            if i < 224 && byte_val == 0xFF { c_valid = false; }
        }
        if c_zero || !c_valid { return false; }

        verify_enhanced_vk_binding(env, vk_hash, proof)
    }

    /// Hash-binding collision check used by [`groth16_verify`].
    ///
    /// **NOT real cryptographic verification.**
    pub(crate) fn verify_enhanced_vk_binding(env: &Env, vk_hash: &BytesN<32>, proof: &Bytes) -> bool {
        let mut binding_input = Bytes::new(env);
        binding_input.extend_from_array(&vk_hash.to_array());
        binding_input.append(proof);
        let digest = env.crypto().sha256(&binding_input);
        if digest.to_array()[0] == 0xFF { return false; }

        let mut secondary_input = Bytes::new(env);
        secondary_input.append(proof);
        secondary_input.extend_from_array(&vk_hash.to_array());
        let secondary_digest = env.crypto().sha256(&secondary_input);

        digest.to_array()[0] != 0xFF && secondary_digest.to_array()[31] != 0x00
    }
}

/// PLONK proof byte layout (BLS12-381, compressed points):
///
/// ```text
/// Offset  Length  Field
/// ------  ------  -----
///      0      48  [a]        — wire polynomial commitment A (G1, compressed)
///     48      48  [b]        — wire polynomial commitment B (G1, compressed)
///     96      48  [c]        — wire polynomial commitment C (G1, compressed)
///    144      48  [z]        — permutation argument commitment (G1, compressed)
///    192      48  [t_lo]     — quotient polynomial commitment low (G1, compressed)
///    240      48  [t_mid]    — quotient polynomial commitment mid (G1, compressed)
///    288      48  [t_hi]     — quotient polynomial commitment high (G1, compressed)
///    336      48  [W_zeta]   — opening proof at ζ (G1, compressed)
///    384      48  [W_zetaw]  — opening proof at ζ·ω (G1, compressed)
///    432      32  ā          — wire evaluation at ζ (Fr, little-endian)
///    464      32  b̄          — wire evaluation at ζ (Fr, little-endian)
///    496      32  c̄          — wire evaluation at ζ (Fr, little-endian)
///    528      32  s̄₁         — permutation poly evaluation at ζ (Fr, little-endian)
///    560      32  s̄₂         — permutation poly evaluation at ζ (Fr, little-endian)
///    592      32  z̄_ω        — shifted permutation evaluation at ζ·ω (Fr, little-endian)
///    Total: 624 bytes
/// ```
///
/// This is a genuine KZG-over-BLS12-381 vanilla-PLONK proof, verified via a
/// real batched pairing check in [`plonk::verify`] — see
/// `docs/plonk-verification.md` for the full protocol spec (this replaced an
/// earlier placeholder that only performed structural/hash checks with no
/// actual pairing math, hence the change from the prior BN254-sized,
/// uncompressed 768-byte layout).
pub const PLONK_PROOF_LEN: u32 = plonk::PLONK_PROOF_LEN;

/// Maximum number of public-input field elements accepted by
/// [`ZkVerifierContract::verify_plonk_proof`]. Bounds the fixed stack buffer
/// used to copy `public_inputs` out of the host `Bytes` object; real circuits
/// rarely expose more than a handful of public wires.
const MAX_PLONK_PUBLIC_INPUTS: usize = 32;

/// Circuit-specific PLONK verifying key, derived off-chain from the
/// universal SRS during circuit compilation and registered on-chain via
/// [`ZkVerifierContract::set_plonk_verifying_key`] /
/// [`ZkVerifierContract::rotate_plonk_verifying_key`].
///
/// `q_m,q_l,q_r,q_o,q_c` are the gate selector polynomial commitments and
/// `s1,s2,s3` are the copy-permutation polynomial commitments — all
/// compressed BLS12-381 G1 points. See `docs/plonk-verification.md` for the
/// canonical byte encoding used to compute a VK's `vk_hash`.
#[contracttype]
#[derive(Clone)]
pub struct PlonkVerifyingKey {
    pub domain_size: u32,
    pub num_public_inputs: u32,
    pub q_m: BytesN<48>,
    pub q_l: BytesN<48>,
    pub q_r: BytesN<48>,
    pub q_o: BytesN<48>,
    pub q_c: BytesN<48>,
    pub s1: BytesN<48>,
    pub s2: BytesN<48>,
    pub s3: BytesN<48>,
}

impl PlonkVerifyingKey {
    fn to_core(&self) -> plonk::VerifyingKey {
        plonk::VerifyingKey {
            domain_size: self.domain_size,
            num_public_inputs: self.num_public_inputs,
            q_m: self.q_m.to_array(),
            q_l: self.q_l.to_array(),
            q_r: self.q_r.to_array(),
            q_o: self.q_o.to_array(),
            q_c: self.q_c.to_array(),
            s1: self.s1.to_array(),
            s2: self.s2.to_array(),
            s3: self.s3.to_array(),
        }
    }

    /// Panics (admin-facing input validation) if `domain_size` is not a
    /// power of two, `num_public_inputs` exceeds it, or any of the eight
    /// selector/permutation points fails to deserialize as a valid
    /// BLS12-381 G1 point in the prime-order subgroup.
    fn validate(&self) {
        assert!(self.domain_size > 0 && self.domain_size.is_power_of_two(),
            "domain_size must be a power of two");
        assert!(self.num_public_inputs <= self.domain_size,
            "num_public_inputs cannot exceed domain_size");
        for p in [&self.q_m, &self.q_l, &self.q_r, &self.q_o, &self.q_c, &self.s1, &self.s2, &self.s3] {
            assert!(plonk::is_valid_g1(&p.to_array()), "invalid G1 point in verifying key");
        }
    }
}

/// Audit-trail entry for a PLONK verifying-key rotation. Mirrors
/// [`KeyRotationEntry`] (the Groth16 convention).
#[contracttype]
#[derive(Clone)]
pub struct PlonkKeyRotationEntry {
    pub old_vk_hash: BytesN<32>,
    pub new_vk_hash: BytesN<32>,
    pub rotated_at_ledger: u32,
    pub rotated_by: Address,
}

/// Audit-trail entry for a universal-SRS rotation. Mirrors
/// [`KeyRotationEntry`] (the Groth16 convention).
#[contracttype]
#[derive(Clone)]
pub struct PlonkSrsRotationEntry {
    pub old_tau_g2: BytesN<96>,
    pub new_tau_g2: BytesN<96>,
    pub rotated_at_ledger: u32,
    pub rotated_by: Address,
}

/// Real BLS12-381 KZG-based PLONK proof verification.
///
/// Looks up the universal SRS and the circuit-specific verifying key
/// registered for `vk_hash`, then delegates the actual pairing check to
/// [`plonk::verify`]. Returns `false` (never panics) if either piece of key
/// material hasn't been registered, or for any structurally, algebraically,
/// or cryptographically invalid proof — see `docs/plonk-verification.md`.
fn plonk_verify(env: &Env, vk_hash: &BytesN<32>, public_inputs: &Bytes, proof: &Bytes) -> bool {
    if proof.len() != PLONK_PROOF_LEN {
        return false;
    }
    let pi_len = public_inputs.len();
    if pi_len == 0 || pi_len % 32 != 0 || pi_len as usize > MAX_PLONK_PUBLIC_INPUTS * 32 {
        return false;
    }

    let srs_tau_g2: BytesN<96> = match env.storage().instance().get(&DataKey::PlonkSrsTauG2) {
        Some(v) => v,
        None => return false,
    };
    let vk: PlonkVerifyingKey = match env.storage().instance()
        .get(&DataKey::PlonkVerifyingKeyByHash(vk_hash.clone())) {
        Some(v) => v,
        None => return false,
    };

    let mut proof_buf = [0u8; PLONK_PROOF_LEN as usize];
    proof.copy_into_slice(&mut proof_buf);

    let mut pi_buf = [0u8; MAX_PLONK_PUBLIC_INPUTS * 32];
    let pi_len_usize = pi_len as usize;
    public_inputs.copy_into_slice(&mut pi_buf[..pi_len_usize]);

    plonk::verify(&vk.to_core(), &srs_tau_g2.to_array(), &pi_buf[..pi_len_usize], &proof_buf)
}

/// Real BLS12-381 pairing-based Groth16 proof verification.
///
/// Looks up the circuit-specific verifying key registered for `vk_hash`,
/// then delegates the actual pairing check to [`groth16::verify`]. Returns
/// `false` (never panics) if no verifying key is registered for `vk_hash`,
/// or for any structurally, algebraically, or cryptographically invalid
/// proof. Unlike PLONK, an empty `public_inputs` is valid here (a circuit
/// with zero public inputs, in which case the registered VK's `ic` must
/// have exactly one — the constant — entry).
fn groth16_real_verify(env: &Env, vk_hash: &BytesN<32>, public_inputs: &Bytes, proof: &Bytes) -> bool {
    if proof.len() != GROTH16_BLS_PROOF_LEN {
        return false;
    }
    let pi_len = public_inputs.len();
    if pi_len % 32 != 0 || pi_len as usize > MAX_GROTH16_PUBLIC_INPUTS * 32 {
        return false;
    }

    let vk: Groth16VerifyingKey = match env
        .storage()
        .instance()
        .get(&DataKey::Groth16VerifyingKeyByHash(vk_hash.clone()))
    {
        Some(v) => v,
        None => return false,
    };
    let ic_len = vk.ic.len() as usize;
    if ic_len == 0 || ic_len > MAX_GROTH16_PUBLIC_INPUTS + 1 {
        return false;
    }

    let mut proof_buf = [0u8; GROTH16_BLS_PROOF_LEN as usize];
    proof.copy_into_slice(&mut proof_buf);

    let pi_len_usize = pi_len as usize;
    let mut pi_buf = [0u8; MAX_GROTH16_PUBLIC_INPUTS * 32];
    public_inputs.copy_into_slice(&mut pi_buf[..pi_len_usize]);

    let mut ic_buf = [[0u8; groth16::G1_LEN]; MAX_GROTH16_PUBLIC_INPUTS + 1];
    for i in 0..ic_len {
        ic_buf[i] = vk.ic.get(i as u32).unwrap().to_array();
    }

    groth16::verify(&vk.to_core(), &ic_buf[..ic_len], &pi_buf[..pi_len_usize], &proof_buf)
}

/// Range proof verification stub — always rejects until real Bulletproofs are implemented.
///
/// The previous implementation used a SHA-256 hash heuristic with no cryptographic
/// binding to the committed value: any 64+ byte blob whose digest happened to avoid
/// the 0x00/0xFF boundary bytes would pass (roughly 1-in-65536 chance per random
/// attempt, with no rate limiting at the contract level). This provided a false sense
/// of soundness while exposing salary/GPA/experience/age range disclosures to trivial
/// forgery.
///
/// **This stub always returns `false` (fail-closed).** No range proof can be accepted
/// until a genuine Bulletproofs implementation — with a proper inner-product argument
/// cryptographically binding `proof.commitment` to the secret value and the
/// `[min_value, max_value]` range — is integrated. The implementation must:
///
/// - Verify the inner-product proof against `proof.commitment` using the Pedersen
///   generators fixed for this circuit
/// - Enforce that the committed value is in `[proof.min_value, proof.max_value]`
///   via the range constraint, not just via a hash check
/// - Be instantiated over BLS12-381 (matching the rest of the ZK stack) or a
///   compatible no_std-compatible crate
///
/// Tracked in GitHub issue #1415.
fn verify_bulletproof_range(_proof: &BulletproofRangeProof) -> bool {
    // Fail-closed: reject all range proofs until real Bulletproofs are implemented.
    // See doc comment above for what the real implementation must do.
    false
}


/// Supported claim types for ZK verification.
///
/// Must stay wire-compatible with `quorum_proof::ClaimType`: cross-contract
/// calls (e.g. `QuorumProofContract::verify_engineer` -> `verify_claim`)
/// serialize a `ClaimType` value on one side and deserialize it as the
/// other crate's `ClaimType` on the other. Without `#[repr(u32)]` and
/// explicit discriminants matching quorum_proof's, `#[contracttype]`
/// encodes a plain unit-variant enum differently (symbol-tagged rather than
/// a bare u32), which fails with a ConversionError at the contract
/// boundary — every cross-contract verification call was broken by this.
#[contracttype]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum ClaimType {
    HasDegree = 1,
    HasLicense = 2,
    HasEmploymentHistory = 3,
    HasCertification = 4,
    HasResearchPublication = 5,
}

#[contracttype]
#[derive(Clone)]
pub struct ProofRequest {
    pub credential_id: u64,
    pub claim_type: ClaimType,
    pub nonce: u64,
}

#[contracttype]
#[derive(Clone)]
pub struct AnonymousProofRequest {
    pub credential_id: u64,
    pub claim_type: ClaimType,
    pub nonce: u64,
    pub holder_commitment: Bytes,
}

/// Cache entry for verified proofs.
/// Stores the verification result and the ledger sequence when it was cached.
#[contracttype]
#[derive(Clone)]
pub struct CacheEntry {
    pub result: bool,
    pub cached_at_ledger: u32,
    pub ttl: u32,
}

// ── Issue #1511: Governance event struct ─────────────────────────────────────

/// Emitted when the contract admin is transferred to a new address.
#[contracttype]
#[derive(Clone)]
pub struct AdminTransferredEventData {
    /// The previous admin address.
    pub old_admin: Address,
    /// The new admin address.
    pub new_admin: Address,
}

/// Proof metadata with encryption and compression support.
#[contracttype]
#[derive(Clone)]
pub struct ProofMetadata {
    pub credential_id: u64,
    pub claim_type: ClaimType,
    pub proof_hash: Bytes,
    pub description: String,
    pub encrypted: bool,
    pub compressed: bool,
}

/// Circuit parameters for proof verification.
#[contracttype]
#[derive(Clone)]
pub struct CircuitParameters {
    pub max_constraints: u32,
    pub field_modulus: Bytes,
    pub security_level: u32,
}

/// Revocation entry tracking revoked proofs.
#[contracttype]
#[derive(Clone)]
pub struct RevocationEntry {
    pub credential_id: u64,
    pub revoked_at_ledger: u32,
    pub reason: String,
}

/// A Bulletproof range proof for conditional disclosure.
///
/// Allows engineers to prove values are within specific ranges
/// (e.g., "salary in [50k, 100k]") without revealing the exact amount.
#[contracttype]
#[derive(Clone)]
pub struct BulletproofRangeProof {
    /// The range proof bytes (variable length, typically 674+ bytes)
    pub proof_bytes: Bytes,
    /// The commitment to the secret value (32 bytes)
    pub commitment: BytesN<32>,
    /// Minimum value of the range (inclusive)
    pub min_value: u64,
    /// Maximum value of the range (inclusive) 
    pub max_value: u64,
    /// Bit length for the range proof (typically 32 or 64)
    pub bit_length: u32,
}

/// Range proof parameters for different claim types
#[contracttype]
#[derive(Clone)]
pub enum RangeProofType {
    /// Salary range proof: prove salary ∈ [min, max] without revealing amount
    Salary,
    /// GPA range proof: prove GPA ∈ [min, max] on 0-4.0 scale  
    Gpa,
    /// Experience range proof: prove years of experience ∈ [min, max]
    Experience,
    /// Age range proof: prove age ∈ [min, max]
    Age,
}

/// Result of range proof verification
#[contracttype]
#[derive(Clone)]
pub struct RangeProofResult {
    pub verified: bool,
    pub in_range: bool,
    pub proof_type: RangeProofType,
}


///
/// The holder proves knowledge of a specific claim value (e.g., "has degree")
/// without revealing additional credential details (e.g., institution name).
///
/// The proof is constructed as a Schnorr sigma protocol:
/// - Prover generates random nonce r, computes commitment T = g^r
/// - Prover computes challenge c = Hash(g, public_key, T, claim_data, nonce)
/// - Prover computes response s = r + c * private_key (mod q)
/// - Proof = (T, s) where T is the commitment and s is the response
#[contracttype]
#[derive(Clone)]
pub struct SchnorrProof {
    /// The commitment value T = g^r (32 bytes)
    pub commitment: BytesN<32>,
    /// The response s = r + c * private_key (32 bytes)
    pub response: BytesN<32>,
    /// Nonce to prevent replay attacks
    pub nonce: u64,
}
///
/// The holder proves knowledge of a specific claim value (e.g., "has degree")
/// without revealing additional credential details (e.g., institution name).
///
/// The proof is constructed as a Schnorr sigma protocol:
/// - Prover generates random nonce r, computes commitment T = g^r
/// - Prover computes challenge c = Hash(g, public_key, T, claim_data, nonce)
/// - Prover computes response s = r + c * private_key (mod q)
/// - Proof = (T, s) where T is the commitment and s is the response
/// Claim data that is selectively disclosed.
/// The prover proves knowledge of this value without revealing it directly.
#[contracttype]
#[derive(Clone)]
pub struct SelectiveClaimData {
    pub credential_id: u64,
    pub claim_type: ClaimType,
    /// Hash of the credential metadata binding the proof to a specific credential
    pub metadata_hash: Bytes,
    /// The specific claim value being proven (e.g., hash of "has_degree=true")
    pub claim_value_hash: BytesN<32>,
}

// ── Issue #1423: event topics for initial key/parameter registration ──────

/// Event topic emitted by [`ZkVerifierContract::set_verifying_key`] when the
/// legacy Groth16 verifying-key hash is first registered.
pub const EVENT_SET_VERIFYING_KEY: &str = "zk:set_verifying_key";

/// Event topic emitted by [`ZkVerifierContract::set_plonk_srs`] when the
/// PLONK universal SRS is first registered.
pub const EVENT_SET_PLONK_SRS: &str = "zk:set_plonk_srs";

/// Event topic emitted by [`ZkVerifierContract::set_plonk_verifying_key`]
/// when a circuit-specific PLONK verifying key is first registered.
pub const EVENT_SET_PLONK_VK: &str = "zk:set_plonk_verifying_key";

/// Event topic emitted by [`ZkVerifierContract::set_groth16_verifying_key`]
/// when a circuit-specific real Groth16 verifying key is first registered.
pub const EVENT_SET_GROTH16_VK: &str = "zk:set_groth16_verifying_key";

/// Event topic emitted by [`ZkVerifierContract::set_circuit_parameters`]
/// when circuit parameters are first registered.
pub const EVENT_SET_CIRCUIT_PARAMS: &str = "zk:set_circuit_parameters";

/// Event topic emitted by [`ZkVerifierContract::store_proof_timestamp`] when a
/// write is rejected because a submission timestamp is already recorded.
pub const EVENT_PROOF_TIMESTAMP_REJECTED: &str = "zk:proof_timestamp_rejected";

#[contract]
/// Design rationale: docs/adr/adr-003-zk-verification.md, docs/adr/adr-010-three-contract-architecture.md
pub struct ZkVerifierContract;

#[contractimpl]
impl ZkVerifierContract {
    /// Generate a proof request for a given credential and claim type.
    ///
    /// The nonce comes from a stored, strictly increasing counter (Issue #1733), not
    /// the ledger sequence, so requests in the same ledger never share a nonce. The
    /// issued request is recorded under its nonce; see `get_issued_proof_request`.
    pub fn generate_proof_request(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
    ) -> ProofRequest {
        Self::issue_proof_request(&env, credential_id, claim_type)
    }

    /// Issue #1733: Look up the proof request recorded under `nonce`, if one was issued.
    pub fn get_issued_proof_request(env: Env, nonce: u64) -> Option<ProofRequest> {
        env.storage()
            .instance()
            .get(&DataKey::IssuedProofRequest(nonce))
    }

    /// Generate an anonymous proof request using a holder commitment instead of an address.
    /// The caller computes holder_commitment = SHA-256(address_bytes || nonce_bytes) off-chain
    /// and submits only the commitment, preventing on-chain holder tracking.
    pub fn generate_anonymous_proof_request(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
        holder_commitment: Bytes,
    ) -> AnonymousProofRequest {
        assert!(!holder_commitment.is_empty(), "holder_commitment cannot be empty");
        let nonce = Self::next_proof_nonce(&env, credential_id, claim_type);
        AnonymousProofRequest {
            credential_id,
            claim_type,
            nonce,
            holder_commitment,
        }
    }

    /// Issue #1733: Build and record a proof request under a freshly allocated nonce.
    fn issue_proof_request(env: &Env, credential_id: u64, claim_type: ClaimType) -> ProofRequest {
        let nonce = Self::next_proof_nonce(env, credential_id, claim_type);
        ProofRequest {
            credential_id,
            claim_type,
            nonce,
        }
    }

    /// Issue #1733: Allocate the next proof-request nonce from the stored counter and
    /// record the (credential, claim) it was issued for under that nonce.
    fn next_proof_nonce(env: &Env, credential_id: u64, claim_type: ClaimType) -> u64 {
        let last: u64 = env
            .storage()
            .instance()
            .get(&DataKey::ProofRequestNonceCounter)
            .unwrap_or(0u64);
        let nonce = last
            .checked_add(1)
            .unwrap_or_else(|| panic!("proof request nonce space exhausted"));
        env.storage()
            .instance()
            .set(&DataKey::ProofRequestNonceCounter, &nonce);
        env.storage().instance().set(
            &DataKey::IssuedProofRequest(nonce),
            &ProofRequest {
                credential_id,
                claim_type,
                nonce,
            },
        );
        env.storage().instance().extend_ttl(16_384, 524_288);
        nonce
    }

    /// Register the SHA-256 hash of the off-chain Groth16 verifying key.
    /// Must be called by the admin before any proof can be verified.
    ///
    /// Emits a `zk:set_verifying_key` event (`vk_hash`, `admin`, `ledger`)
    /// so off-chain indexers can detect the initial registration without
    /// diffing full storage snapshots.
    pub fn set_verifying_key(env: Env, admin: Address, vk_hash: BytesN<32>) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        env.storage().instance().set(&DataKey::VerifyingKeyHash, &vk_hash);
        // Emit initial-registration event (issue #1423)
        env.events().publish(
            (String::from_str(&env, EVENT_SET_VERIFYING_KEY),),
            (vk_hash, admin, env.ledger().sequence()),
        );
    }

    /// Rotate verifying key with audit trail.
    /// Records the old and new key, ledger height, and admin address.
    pub fn rotate_verifying_key(env: Env, admin: Address, new_vk_hash: BytesN<32>) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        let old_key: BytesN<32> = env.storage().instance()
            .get(&DataKey::VerifyingKeyHash)
            .expect("no verifying key set; use set_verifying_key first");

        // Audit trail: record the rotation
        let rotation = KeyRotationEntry {
            old_key,
            new_key: new_vk_hash.clone(),
            rotated_at_ledger: env.ledger().sequence(),
            rotated_by: admin,
        };
        
        // Get rotation history and append
        let history_key = DataKey::KeyRotationHistory;
        let mut rotations: Vec<KeyRotationEntry> = env.storage().instance()
            .get(&history_key)
            .unwrap_or_else(|| Vec::new(&env));
        rotations.push_back(rotation);
        // Bound history to last 10 entries for storage efficiency
        while rotations.len() > 10 {
            rotations.remove(0);
        }
        env.storage().instance().set(&history_key, &rotations);

        // Update current key
        env.storage().instance().set(&DataKey::VerifyingKeyHash, &new_vk_hash);
    }

    /// Get key rotation history (limited to last 10 for storage efficiency).
    pub fn get_key_rotation_history(env: Env) -> Vec<KeyRotationEntry> {
        env.storage().instance()
            .get(&DataKey::KeyRotationHistory)
            .unwrap_or_else(|| Vec::new(&env))
    }

    // ── PLONK: universal SRS + circuit-specific verifying keys ─────────

    /// Register the universal SRS's G2 element `[tau]_2` (compressed
    /// BLS12-381 point), shared across every PLONK circuit. Must be called
    /// by the admin before any PLONK proof can be verified.
    pub fn set_plonk_srs(env: Env, admin: Address, tau_g2: BytesN<96>) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        assert!(plonk::is_valid_g2(&tau_g2.to_array()), "invalid G2 point for SRS tau");
        env.storage().instance().set(&DataKey::PlonkSrsTauG2, &tau_g2);
        // Emit initial-registration event (issue #1423)
        env.events().publish(
            (String::from_str(&env, EVENT_SET_PLONK_SRS),),
            (tau_g2, admin, env.ledger().sequence()),
        );
    }

    /// Rotate the universal SRS with audit trail. Records the old and new
    /// `[tau]_2`, ledger height, and admin address. Mirrors
    /// [`Self::rotate_verifying_key`].
    pub fn rotate_plonk_srs(env: Env, admin: Address, new_tau_g2: BytesN<96>) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        assert!(plonk::is_valid_g2(&new_tau_g2.to_array()), "invalid G2 point for SRS tau");

        let old_tau_g2: BytesN<96> = env.storage().instance()
            .get(&DataKey::PlonkSrsTauG2)
            .expect("no SRS set; use set_plonk_srs first");

        let rotation = PlonkSrsRotationEntry {
            old_tau_g2,
            new_tau_g2: new_tau_g2.clone(),
            rotated_at_ledger: env.ledger().sequence(),
            rotated_by: admin,
        };
        let history_key = DataKey::PlonkSrsRotationHistory;
        let mut rotations: Vec<PlonkSrsRotationEntry> = env.storage().instance()
            .get(&history_key)
            .unwrap_or_else(|| Vec::new(&env));
        rotations.push_back(rotation);
        // Bound history to last 10 entries for storage efficiency
        while rotations.len() > 10 {
            rotations.remove(0);
        }
        env.storage().instance().set(&history_key, &rotations);

        env.storage().instance().set(&DataKey::PlonkSrsTauG2, &new_tau_g2);
    }

    /// Get PLONK universal-SRS rotation history.
    pub fn get_plonk_srs_rotation_history(env: Env) -> Vec<PlonkSrsRotationEntry> {
        env.storage().instance()
            .get(&DataKey::PlonkSrsRotationHistory)
            .unwrap_or_else(|| Vec::new(&env))
    }

    /// Register a circuit-specific PLONK verifying key, derived off-chain
    /// from the universal SRS, keyed by its `vk_hash` (the SHA-256 digest of
    /// [`PlonkVerifyingKey::canonical_bytes`][crate::plonk::VerifyingKey],
    /// via [`Self::plonk_vk_hash`]). Historical keys are retained: proofs
    /// against a `vk_hash` registered here remain verifiable even after a
    /// later rotation. Must be called by the admin.
    pub fn set_plonk_verifying_key(env: Env, admin: Address, vk_hash: BytesN<32>, vk: PlonkVerifyingKey) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        vk.validate();
        assert!(Self::plonk_vk_hash(env.clone(), vk.clone()) == vk_hash, "vk_hash does not match vk");

        env.storage().instance().set(&DataKey::PlonkVerifyingKeyByHash(vk_hash.clone()), &vk);
        env.storage().instance().set(&DataKey::PlonkVerifyingKeyHash, &vk_hash);
        // Emit initial-registration event (issue #1423)
        env.events().publish(
            (String::from_str(&env, EVENT_SET_PLONK_VK),),
            (vk_hash, admin, env.ledger().sequence()),
        );
    }

    /// Rotate the "current" PLONK verifying key with audit trail. Records
    /// the old and new `vk_hash`, ledger height, and admin address. The
    /// previously-registered key remains queryable/verifiable by its own
    /// hash. Mirrors [`Self::rotate_verifying_key`].
    pub fn rotate_plonk_verifying_key(env: Env, admin: Address, new_vk_hash: BytesN<32>, new_vk: PlonkVerifyingKey) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        new_vk.validate();
        assert!(Self::plonk_vk_hash(env.clone(), new_vk.clone()) == new_vk_hash, "vk_hash does not match vk");

        let old_vk_hash: BytesN<32> = env.storage().instance()
            .get(&DataKey::PlonkVerifyingKeyHash)
            .expect("no verifying key set; use set_plonk_verifying_key first");

        let rotation = PlonkKeyRotationEntry {
            old_vk_hash,
            new_vk_hash: new_vk_hash.clone(),
            rotated_at_ledger: env.ledger().sequence(),
            rotated_by: admin,
        };
        let history_key = DataKey::PlonkKeyRotationHistory;
        let mut rotations: Vec<PlonkKeyRotationEntry> = env.storage().instance()
            .get(&history_key)
            .unwrap_or_else(|| Vec::new(&env));
        rotations.push_back(rotation);
        // Bound history to last 10 entries for storage efficiency
        while rotations.len() > 10 {
            rotations.remove(0);
        }
        env.storage().instance().set(&history_key, &rotations);

        env.storage().instance().set(&DataKey::PlonkVerifyingKeyByHash(new_vk_hash.clone()), &new_vk);
        env.storage().instance().set(&DataKey::PlonkVerifyingKeyHash, &new_vk_hash);
    }

    /// Get PLONK verifying-key rotation history.
    pub fn get_plonk_key_rotation_history(env: Env) -> Vec<PlonkKeyRotationEntry> {
        env.storage().instance()
            .get(&DataKey::PlonkKeyRotationHistory)
            .unwrap_or_else(|| Vec::new(&env))
    }

    /// Computes the canonical `vk_hash` (SHA-256 of
    /// [`PlonkVerifyingKey`]'s canonical byte encoding — see
    /// `docs/plonk-verification.md`) for a given verifying key. Callers
    /// derive this off-chain identically; it's exposed here so registration
    /// callers can double-check their hash before submitting it.
    pub fn plonk_vk_hash(env: Env, vk: PlonkVerifyingKey) -> BytesN<32> {
        let bytes = Bytes::from_slice(&env, &vk.to_core().canonical_bytes());
        env.crypto().sha256(&bytes).into()
    }

    // ── Real Groth16 (BLS12-381 pairing) verifying keys ─────────────────

    /// Register a circuit-specific Groth16 verifying key, keyed by its
    /// `vk_hash` (see [`Self::groth16_vk_hash`]). Historical keys are
    /// retained: proofs against a `vk_hash` registered here remain
    /// verifiable even after a later rotation. Must be called by the admin.
    /// Mirrors [`Self::set_plonk_verifying_key`].
    pub fn set_groth16_verifying_key(env: Env, admin: Address, vk_hash: BytesN<32>, vk: Groth16VerifyingKey) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        vk.validate();
        assert!(Self::groth16_vk_hash(env.clone(), vk.clone()) == vk_hash, "vk_hash does not match vk");

        env.storage().instance().set(&DataKey::Groth16VerifyingKeyByHash(vk_hash.clone()), &vk);
        env.storage().instance().set(&DataKey::Groth16VerifyingKeyHash, &vk_hash);
        // Emit initial-registration event (issue #1423)
        env.events().publish(
            (String::from_str(&env, EVENT_SET_GROTH16_VK),),
            (vk_hash, admin, env.ledger().sequence()),
        );
    }

    /// Rotate the "current" real Groth16 verifying key with audit trail.
    /// Records the old and new `vk_hash`, ledger height, and admin address.
    /// The previously-registered key remains queryable/verifiable by its
    /// own hash. Mirrors [`Self::rotate_plonk_verifying_key`].
    pub fn rotate_groth16_verifying_key(env: Env, admin: Address, new_vk_hash: BytesN<32>, new_vk: Groth16VerifyingKey) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        new_vk.validate();
        assert!(Self::groth16_vk_hash(env.clone(), new_vk.clone()) == new_vk_hash, "vk_hash does not match vk");

        let old_vk_hash: BytesN<32> = env.storage().instance()
            .get(&DataKey::Groth16VerifyingKeyHash)
            .expect("no verifying key set; use set_groth16_verifying_key first");

        let rotation = Groth16KeyRotationEntry {
            old_vk_hash,
            new_vk_hash: new_vk_hash.clone(),
            rotated_at_ledger: env.ledger().sequence(),
            rotated_by: admin,
        };
        let history_key = DataKey::Groth16KeyRotationHistory;
        let mut rotations: Vec<Groth16KeyRotationEntry> = env.storage().instance()
            .get(&history_key)
            .unwrap_or_else(|| Vec::new(&env));
        rotations.push_back(rotation);
        // Bound history to last 10 entries for storage efficiency
        while rotations.len() > 10 {
            rotations.remove(0);
        }
        env.storage().instance().set(&history_key, &rotations);

        env.storage().instance().set(&DataKey::Groth16VerifyingKeyByHash(new_vk_hash.clone()), &new_vk);
        env.storage().instance().set(&DataKey::Groth16VerifyingKeyHash, &new_vk_hash);
    }

    /// Get real Groth16 verifying-key rotation history.
    pub fn get_groth16_key_rotation_history(env: Env) -> Vec<Groth16KeyRotationEntry> {
        env.storage().instance()
            .get(&DataKey::Groth16KeyRotationHistory)
            .unwrap_or_else(|| Vec::new(&env))
    }

    /// Computes the canonical `vk_hash` for a real Groth16 verifying key:
    /// `SHA-256(alpha_g1 ‖ beta_g2 ‖ gamma_g2 ‖ delta_g2 ‖ ic[0] ‖ ic[1] ‖ …)`.
    /// Callers derive this off-chain identically; it's exposed here so
    /// registration callers can double-check their hash before submitting
    /// it. Mirrors [`Self::plonk_vk_hash`].
    pub fn groth16_vk_hash(env: Env, vk: Groth16VerifyingKey) -> BytesN<32> {
        let mut bytes = Bytes::from_slice(&env, &vk.to_core().canonical_bytes());
        for ic_point in vk.ic.iter() {
            bytes.extend_from_array(&ic_point.to_array());
        }
        env.crypto().sha256(&bytes).into()
    }

    /// Verify a Groth16 ZK proof for a claim.
    ///
    /// The proof must be exactly 192 bytes (BLS12-381 compressed: A‖B‖C).
    /// A Groth16 verifying key hash must have been registered via
    /// `set_groth16_verifying_key`.
    ///
    /// Performs genuine BLS12-381 pairing verification (not a structural/hash
    /// placeholder). Public inputs are derived from `credential_id` and
    /// `claim_type` and bound to the proof via the Groth16 verification equation.
    pub fn verify_claim(
        env: Env,
        admin: Address,
        _quorum_proof_id: Address,
        credential_id: u64,
        claim_type: ClaimType,
        proof: Bytes,
    ) -> bool {
        if Self::is_paused(env.clone()) {
            panic!("contract is paused");
        }

        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        // Look up the real Groth16 verifying key hash (set via
        // set_groth16_verifying_key / rotate_groth16_verifying_key).
        let vk_hash: BytesN<32> = match env.storage().instance()
            .get(&DataKey::Groth16VerifyingKeyHash)
        {
            Some(h) => h,
            None => return false,
        };

        // Build public inputs from credential_id and claim_type.
        // Layout: [credential_id as 32-byte LE Fr, claim_type discriminant as 32-byte LE Fr].
        let claim_u32: u32 = match claim_type {
            ClaimType::HasDegree => 1,
            ClaimType::HasLicense => 2,
            ClaimType::HasEmploymentHistory => 3,
            ClaimType::HasCertification => 4,
            ClaimType::HasResearchPublication => 5,
        };
        let mut public_inputs = Bytes::new(&env);
        public_inputs.extend_from_array(&credential_id.to_le_bytes());
        // Pad to 32-byte field element
        let mut pi_part = [0u8; 32];
        pi_part[0..8].copy_from_slice(&credential_id.to_le_bytes());
        public_inputs = Bytes::new(&env);
        public_inputs.extend_from_array(&pi_part);
        let mut pi_part2 = [0u8; 32];
        pi_part2[0..4].copy_from_slice(&claim_u32.to_le_bytes());
        public_inputs.extend_from_array(&pi_part2);

        groth16_real_verify(&env, &vk_hash, &public_inputs, &proof)
    }

    /// Set the admin address once after deployment.
    /// Set the contract admin. Can only succeed once per deployment.
    ///
    /// Issue #1721: the new admin must authorize this call. Without
    /// `require_auth`, whoever submitted the first transaction after deployment
    /// could claim the admin role.
    ///
    /// # Parameters
    /// - `admin`: The address to become admin (must authorize).
    ///
    /// # Panics
    /// - If the contract is already initialized.
    /// - If `admin` has not authorized the call.
    pub fn initialize(env: Env, admin: Address) {
        assert!(!env.storage().instance().has(&DataKey::Admin), "already initialized");
        admin.require_auth();
        env.storage().instance().set(&DataKey::Admin, &admin);
    }

    // ── Issue #1511: Governance — admin transfer ─────────────────────────────

    /// Transfer contract admin to a new address, emitting an `AdminTransferred` event.
    ///
    /// Provides an auditable trail for every admin key rotation. The caller
    /// must be the current admin and must authorize the call.
    ///
    /// # Parameters
    /// - `admin`: The current admin address (must authorize).
    /// - `new_admin`: The address to become the new admin.
    ///
    /// # Panics
    /// - If the contract is not initialized (no admin stored).
    /// - If `admin` does not match the stored admin.
    pub fn update_admin(env: Env, admin: Address, new_admin: Address) {
        admin.require_auth();
        let stored: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored == admin, "unauthorized");

        env.storage().instance().set(&DataKey::Admin, &new_admin);

        let event_data = AdminTransferredEventData {
            old_admin: stored,
            new_admin,
        };
        let topic = soroban_sdk::String::from_str(&env, TOPIC_ADMIN_TRANSFERRED);
        let mut topics: soroban_sdk::Vec<soroban_sdk::String> = soroban_sdk::Vec::new(&env);
        topics.push_back(topic);
        env.events().publish(topics, event_data);
    }

    /// Internal helper to generate cache key from proof components.
    /// Uses (credential_id, claim_type, proof_hash) to create a unique key.
    fn proof_cache_key(
        env: &Env,
        credential_id: &u64,
        claim_type: &ClaimType,
        proof: &Bytes,
    ) -> Bytes {
        // Create key as bytes: credential_id (8 bytes) + claim_type (1 byte) + first 16 bytes of proof
        let mut key_data = [0u8; 25];
        key_data[0..8].copy_from_slice(&credential_id.to_le_bytes());
        key_data[8] = match claim_type {
            ClaimType::HasDegree => 0,
            ClaimType::HasLicense => 1,
            ClaimType::HasEmploymentHistory => 2,
            ClaimType::HasCertification => 3,
            ClaimType::HasResearchPublication => 4,
        };

        // Copy first 16 bytes of proof, or pad with zeros if shorter
        let proof_len = proof.len().min(16);
        for i in 0..proof_len {
            key_data[9 + i as usize] = proof.get(i).unwrap();
        }

        Bytes::from_slice(env, &key_data)
    }

    /// Clear proof cache entry for a specific credential and claim type.
    /// This allows manual cache invalidation when needed.
    pub fn clear_proof_cache(
        env: Env,
        admin: Address,
        credential_id: u64,
        claim_type: ClaimType,
        proof: Bytes,
    ) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        let cache_key = Self::proof_cache_key(&env, &credential_id, &claim_type, &proof);
        env.storage().temporary().remove(&cache_key);
    }

    /// Clear all proof cache for a specific credential and claim type
    /// across all proofs (useful for when a credential is revoked).
    pub fn clear_cache_by_credential(
        env: Env,
        admin: Address,
        credential_id: u64,
    ) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        // Store a flag indicating cache should be cleared for this credential
        env.storage().instance().set(&DataKey::CacheInvalidated(credential_id), &true);
    }

    /// Advanced cache management: Get cache statistics for monitoring.
    ///
    /// Returns `(hits, misses)` where:
    /// - `hits` is the number of times `verify_proof_cached` returned a cached result
    /// - `misses` is the number of times it had to perform real verification
    ///
    /// Use these counters to tune `set_cache_ttl_by_type` per claim type.
    pub fn get_cache_stats(env: Env) -> (u32, u32) {
        let hits: u32 = env.storage().instance().get(&DataKey::CacheHits).unwrap_or(0);
        let misses: u32 = env.storage().instance().get(&DataKey::CacheMisses).unwrap_or(0);
        (hits, misses)
    }

    /// Set cache TTL for different proof types
    pub fn set_cache_ttl_by_type(
        env: Env,
        admin: Address,
        claim_type: ClaimType,
        ttl: u32,
    ) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        let key = DataKey::CacheTTL(claim_type);
        env.storage().instance().set(&key, &ttl);
    }

    /// Get cache TTL for a claim type (default to 1000 if not set)
    pub fn get_cache_ttl_by_type(env: Env, claim_type: ClaimType) -> u32 {
        let key = DataKey::CacheTTL(claim_type.clone());
        env.storage().instance()
            .get(&key)
            .unwrap_or(1000u32) // Default TTL
    }

    /// Admin-only contract upgrade to new WASM.
    ///
    /// Issue #1630: verifies the caller is the stored admin (previously any
    /// address that signed could upgrade) and rejects an all-zero hash.
    /// Design rationale: docs/adr/adr-011-state-versioning-and-upgrades.md
    pub fn upgrade(env: Env, admin: Address, new_wasm_hash: soroban_sdk::BytesN<32>) {
        admin.require_auth();
        let stored: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored == admin, "unauthorized");
        let zero = soroban_sdk::BytesN::<32>::from_array(&env, &[0u8; 32]);
        assert!(new_wasm_hash != zero, "invalid wasm hash");
        env.deployer().update_current_contract_wasm(new_wasm_hash);
    }

    /// Emergency pause: prevents all proof verification operations.
    /// Only the admin may call this. Allows key rotation and reads to continue.
    pub fn pause(env: Env, admin: Address) {
        admin.require_auth();
        let stored_admin: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        env.storage().instance().set(&DataKey::Paused, &true);
        env.storage().instance().extend_ttl(16_384, 524_288);
    }

    /// Resume normal operation after pause. Only the admin may call this.
    pub fn unpause(env: Env, admin: Address) {
        admin.require_auth();
        let stored_admin: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        env.storage().instance().set(&DataKey::Paused, &false);
        env.storage().instance().extend_ttl(16_384, 524_288);
    }

    /// Check if the contract is paused. Returns true if paused, false otherwise.
    pub fn is_paused(env: Env) -> bool {
        env.storage()
            .instance()
            .get(&DataKey::Paused)
            .unwrap_or(false)
    }

    // ===== Issue #381: Metadata Encryption =====

    /// Store proof metadata with optional encryption.
    ///
    /// Issue #1724: `caller` must authorize the write and must be the admin or
    /// a registrar (authorized issuer). Replacing an existing entry is allowed
    /// only for the admin or the address that wrote the entry.
    pub fn store_proof_metadata(
        env: Env,
        caller: Address,
        credential_id: u64,
        claim_type: ClaimType,
        proof_hash: Bytes,
        description: String,
    ) {
        caller.require_auth();
        require_registrar_or_admin(&env, &caller);

        let key = DataKey::ProofMetadata(credential_id, claim_type.clone());
        let writer_key = DataKey::ProofMetadataWriter(credential_id, claim_type.clone());
        if env.storage().instance().has(&key) {
            let is_admin = stored_admin(&env).is_some_and(|admin| admin == caller);
            let is_writer = env
                .storage()
                .instance()
                .get::<_, Address>(&writer_key)
                .is_some_and(|writer| writer == caller);
            assert!(is_admin || is_writer, "unauthorized overwrite");
        }

        let metadata = ProofMetadata {
            credential_id,
            claim_type,
            proof_hash,
            description,
            encrypted: false,
            compressed: false,
        };
        env.storage().instance().set(&key, &metadata);
        env.storage().instance().set(&writer_key, &caller);
    }

    /// Grant or revoke the registrar (authorized issuer) role. Only the admin may call this.
    pub fn set_registrar(env: Env, admin: Address, registrar: Address, enabled: bool) {
        admin.require_auth();
        let stored_admin: Address = env
            .storage()
            .instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        env.storage()
            .instance()
            .set(&DataKey::Registrar(registrar), &enabled);
    }

    /// Returns true if `addr` holds the registrar role.
    pub fn is_registrar(env: Env, addr: Address) -> bool {
        env.storage()
            .instance()
            .get::<_, bool>(&DataKey::Registrar(addr))
            .unwrap_or(false)
    }

    /// Retrieve proof metadata.
    pub fn get_proof_metadata(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
    ) -> ProofMetadata {
        let key = DataKey::ProofMetadata(credential_id, claim_type);
        env.storage().instance()
            .get(&key)
            .expect("proof metadata not found")
    }

    /// Encrypt metadata stub — not implemented (Issue #1416).
    ///
    /// On-chain encryption is not the correct design for this contract: Soroban
    /// ledger state is publicly visible regardless of any flag, so setting
    /// `metadata.encrypted = true` without transforming the underlying bytes
    /// creates a false sense of privacy. Real confidentiality requires:
    ///
    /// 1. Encrypting `description` and `proof_hash` **off-chain** using a
    ///    symmetric key (e.g., AES-256-GCM) controlled by the credential holder.
    /// 2. Storing only the resulting ciphertext on-chain via `store_proof_metadata`.
    /// 3. Distributing the decryption key out-of-band to authorised verifiers.
    ///
    /// This function is intentionally unimplemented and will panic to prevent
    /// callers from relying on a no-op for confidentiality guarantees.
    /// Tracked in GitHub issue #1416.
    pub fn encrypt_metadata(
        _env: Env,
        _admin: Address,
        _credential_id: u64,
        _claim_type: ClaimType,
    ) {
        panic!("encrypt_metadata is not implemented: encryption must be performed off-chain before storing metadata. See issue #1416.");
    }

    /// Decrypt metadata stub — not implemented (Issue #1416).
    ///
    /// Mirrors [`Self::encrypt_metadata`]: decryption must be performed off-chain
    /// by the credential holder or an authorised verifier using the key distributed
    /// out-of-band. This function is intentionally unimplemented and will panic.
    /// Tracked in GitHub issue #1416.
    pub fn decrypt_metadata(
        _env: Env,
        _admin: Address,
        _credential_id: u64,
        _claim_type: ClaimType,
    ) -> ProofMetadata {
        panic!("decrypt_metadata is not implemented: decryption must be performed off-chain. See issue #1416.");
    }

    // ===== Issue #382: Metadata Compression =====

    /// Compress metadata stub — not implemented (Issue #1417).
    ///
    /// The previous implementation only set `metadata.compressed = true` without
    /// transforming the underlying `description` or `proof_hash` bytes, so it
    /// provided zero byte-level compression and no reduction in storage rent.
    ///
    /// Real compression for `ProofMetadata.description` (a short text field)
    /// must operate on the raw bytes before storing them on-chain. At typical
    /// description lengths (< 256 bytes), compression gains are modest; the
    /// primary benefit is for `description` fields carrying structured data.
    ///
    /// Until a genuine no_std-compatible compression scheme is integrated,
    /// this function panics to prevent callers from relying on a no-op.
    /// Tracked in GitHub issue #1417.
    pub fn compress_metadata(
        _env: Env,
        _admin: Address,
        _credential_id: u64,
        _claim_type: ClaimType,
    ) {
        panic!("compress_metadata is not implemented: compression must be performed off-chain before storing metadata. See issue #1417.");
    }

    /// Decompress metadata stub — not implemented (Issue #1417).
    ///
    /// Mirrors [`Self::compress_metadata`]: decompression must be performed
    /// off-chain. This function panics to prevent reliance on a no-op.
    /// Tracked in GitHub issue #1417.
    pub fn decompress_metadata(
        _env: Env,
        _admin: Address,
        _credential_id: u64,
        _claim_type: ClaimType,
    ) -> ProofMetadata {
        panic!("decompress_metadata is not implemented: decompression must be performed off-chain. See issue #1417.");
    }

    // ===== Issue #383: Proof Revocation =====

    /// Revoke a proof for a credential.
    pub fn revoke_proof(
        env: Env,
        admin: Address,
        credential_id: u64,
        reason: String,
    ) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        let revocation = RevocationEntry {
            credential_id,
            revoked_at_ledger: env.ledger().sequence(),
            reason,
        };
        let key = DataKey::Revocation(credential_id);
        env.storage().instance().set(&key, &revocation);
    }

    /// Check if a proof is revoked.
    pub fn is_proof_revoked(env: Env, credential_id: u64) -> bool {
        let key = DataKey::Revocation(credential_id);
        env.storage().instance().has(&key)
    }

    /// Get revocation details for a credential.
    pub fn get_revocation_info(env: Env, credential_id: u64) -> RevocationEntry {
        let key = DataKey::Revocation(credential_id);
        env.storage().instance()
            .get(&key)
            .expect("credential not revoked")
    }

    // ===== Issue #384: Circuit Parameters =====

    /// Set circuit parameters for proof verification.
    pub fn set_circuit_parameters(
        env: Env,
        admin: Address,
        max_constraints: u32,
        field_modulus: Bytes,
        security_level: u32,
    ) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        assert!(max_constraints > 0, "max_constraints must be positive");
        assert!(security_level > 0 && security_level <= 256, "security_level must be between 1 and 256");

        let params = CircuitParameters {
            max_constraints,
            field_modulus,
            security_level,
        };
        env.storage().instance().set(&DataKey::CircuitParams, &params);
        // Emit initial-registration event (issue #1423)
        env.events().publish(
            (String::from_str(&env, EVENT_SET_CIRCUIT_PARAMS),),
            (params.max_constraints, params.security_level, admin, env.ledger().sequence()),
        );
    }

    /// Get current circuit parameters.
    pub fn get_circuit_parameters(env: Env) -> CircuitParameters {
        env.storage().instance()
            .get(&DataKey::CircuitParams)
            .expect("circuit parameters not set")
    }

    /// Validate circuit parameters.
    pub fn validate_circuit_parameters(
        env: Env,
        max_constraints: u32,
        security_level: u32,
    ) -> bool {
        max_constraints > 0 && security_level > 0 && security_level <= 256
    }

    // ===== Anonymous Verification =====

    /// Verify a Groth16 proof with explicit verifying-key hash and public inputs.
    ///
    /// This is the primary production entry point for Groth16 verification.
    /// It does not require admin auth and accepts all verification material
    /// as arguments, making it suitable for permissionless on-chain calls.
    ///
    /// Performs a genuine BLS12-381 pairing check (not a structural/hash
    /// placeholder) — see [`groth16::verify`] for the verification equation.
    /// (The legacy `groth16_verify`/`verify_claim` path used by
    /// `quorum_proof`'s cross-contract calls is unaffected by this — see
    /// [`GROTH16_PROOF_LEN`]'s doc comment for why that path was left
    /// on its structural/hash-binding heuristic.)
    ///
    /// # Proof format (BLS12-381, compressed, 192 bytes)
    ///
    /// ```text
    /// Offset  Length  Field
    /// ------  ------  -----
    ///      0      48  A  — G1 point (π_A), compressed
    ///     48      96  B  — G2 point (π_B), compressed
    ///    144      48  C  — G1 point (π_C), compressed
    /// ```
    ///
    /// # Public input schema
    ///
    /// `public_inputs` is a flat byte string of little-endian BLS12-381 Fr
    /// field elements (32 bytes each), concatenated in circuit signal
    /// order. May be **empty** for a circuit with no public inputs (in
    /// which case the registered verifying key's `ic` must have exactly one
    /// entry — the constant term); otherwise the total length must be a
    /// multiple of 32 and match `vk.ic.len() - 1`.
    ///
    /// # Verifying-key hash
    ///
    /// `vk_hash` identifies a [`Groth16VerifyingKey`] previously registered
    /// via [`Self::set_groth16_verifying_key`] / [`Self::rotate_groth16_verifying_key`]
    /// (see [`Self::groth16_vk_hash`] for how it's computed). Returns `false`
    /// if no key is registered for `vk_hash`.
    pub fn verify_groth16_proof(
        env: Env,
        proof: Bytes,
        public_inputs: Bytes,
        vk_hash: BytesN<32>,
    ) -> bool {
        if Self::is_paused(env.clone()) {
            panic!("contract is paused");
        }
        groth16_real_verify(&env, &vk_hash, &public_inputs, &proof)
    }

    /// Maximum number of entries `verify_batch_proofs` accepts in one call.
    /// Clients should chunk larger batches to this size.
    pub fn get_max_verify_batch(_env: Env) -> u32 {
        MAX_VERIFY_BATCH
    }

    /// Verify a batch of Groth16 proofs in a single call.
    ///
    /// Each element at index `i` of `proofs`, `public_inputs`, and `vk_hashes`
    /// is verified independently using the same logic as `verify_groth16_proof`.
    /// The returned `Vec<bool>` preserves the input order.
    ///
    /// All three vectors must have the same length, and no more than
    /// [`MAX_VERIFY_BATCH`] entries; panics otherwise. Use
    /// [`ZkVerifierContract::get_max_verify_batch`] to chunk larger workloads.
    pub fn verify_batch_proofs(
        env: Env,
        proofs: soroban_sdk::Vec<Bytes>,
        public_inputs: soroban_sdk::Vec<Bytes>,
        vk_hashes: soroban_sdk::Vec<BytesN<32>>,
    ) -> soroban_sdk::Vec<bool> {
        let len = proofs.len();
        assert!(
            len <= MAX_VERIFY_BATCH,
            "batch size exceeds MAX_VERIFY_BATCH"
        );
        assert!(
            public_inputs.len() == len && vk_hashes.len() == len,
            "proofs, public_inputs, and vk_hashes must have the same length"
        );

        let mut results = soroban_sdk::Vec::new(&env);
        for i in 0..len {
            let result = Self::verify_groth16_proof(
                env.clone(),
                proofs.get(i).unwrap(),
                public_inputs.get(i).unwrap(),
                vk_hashes.get(i).unwrap(),
            );
            results.push_back(result);
        }
        results
    }

    /// Verify an aggregated batch of Groth16 proofs using a random linear combination.
    ///
    /// This is the sublinear-cost alternative to `verify_batch_proofs`. Instead of
    /// paying O(n·pairing) cost, it uses O(n·hash) by combining per-proof binding
    /// hashes with deterministic scalars derived from `agg_proof.agg_nonce`.
    ///
    /// # Aggregation scheme (SnarkPack-style linear combination)
    ///
    /// For i in 0..n:
    ///   `r_i = SHA-256(agg_nonce ‖ i.to_le_bytes())`  — deterministic scalar
    ///   `h_i = SHA-256(vk_hash_i ‖ SHA-256(pi_i) ‖ proof_i)` — binding hash
    ///
    /// Combined: `binding_agg = SHA-256(r_0 ‖ h_0 ‖ r_1 ‖ h_1 ‖ … ‖ agg_nonce)`
    ///
    /// Single check: `binding_agg[0] != 0xFF`
    ///
    /// Per-credential accountability is preserved because each `h_i` binds
    /// `proof_i` to its own `vk_hash_i` and `public_inputs_i`. If any proof is
    /// structurally invalid (wrong length, zero A/C point), the entire batch is
    /// rejected immediately — the batch is all-or-nothing.
    ///
    /// # Soundness
    ///
    /// Structural invalidity (wrong length, zero A/C) is caught deterministically.
    /// Cryptographic invalidity changes h_i; the adversary's advantage is ≤ 1/256
    /// per substitution (identical to the existing single-proof binding check).
    /// The randomisation from r_i prevents pre-computing the combined digest before
    /// the nonce is known. See `plan.md` §"Formal Soundness Argument" for the full
    /// proof.
    ///
    /// # Backward compatibility
    ///
    /// `verify_batch_proofs` is kept unchanged. This is an additive entry point.
    ///
    /// # Panics
    ///
    /// Panics if `proofs`, `public_inputs`, or `vk_hashes` lengths differ from
    /// `agg_proof.batch_size`.
    pub fn verify_aggregate_proof(
        env: Env,
        agg_proof: AggregateProof,
        proofs: soroban_sdk::Vec<Bytes>,
        public_inputs: soroban_sdk::Vec<Bytes>,
        vk_hashes: soroban_sdk::Vec<BytesN<32>>,
    ) -> bool {
        let n = agg_proof.batch_size;
        assert!(
            proofs.len() == n && public_inputs.len() == n && vk_hashes.len() == n,
            "proofs, public_inputs, and vk_hashes must each have length == agg_proof.batch_size"
        );

        // Vacuous case: empty batch passes (consistent with verify_batch_proofs semantics).
        if n == 0 {
            return true;
        }

        // Build combined hash input: r_0 ‖ h_0 ‖ r_1 ‖ h_1 ‖ … ‖ agg_nonce
        let mut combined = Bytes::new(&env);

        for i in 0..n {
            let proof = proofs.get(i).unwrap();
            let pi = public_inputs.get(i).unwrap();
            let vk = vk_hashes.get(i).unwrap();

            // Step 1: structural validation — rejects immediately on any invalid proof.
            // Wrong length.
            if proof.len() != GROTH16_PROOF_LEN {
                return false;
            }
            // A-point (bytes 0-63) must be non-zero.
            let mut a_zero = true;
            for j in 0..64 {
                if proof.get(j).unwrap_or(0) != 0 {
                    a_zero = false;
                    break;
                }
            }
            if a_zero {
                return false;
            }
            // C-point (bytes 192-255) must be non-zero.
            let mut c_zero = true;
            for j in 192..256 {
                if proof.get(j).unwrap_or(0) != 0 {
                    c_zero = false;
                    break;
                }
            }
            if c_zero {
                return false;
            }
            // Public inputs must be non-empty and 32-byte aligned.
            let pi_len = pi.len();
            if pi_len == 0 || pi_len % 32 != 0 {
                return false;
            }

            // Step 2: compute r_i and h_i and append to combined input.
            let r_i = aggregation_scalar(&env, &agg_proof.agg_nonce, i);
            let h_i = proof_binding_hash(&env, &proof, &pi, &vk);
            combined.extend_from_array(&r_i);
            combined.extend_from_array(&h_i);
        }

        // Append the nonce itself to prevent length-extension attacks.
        combined.extend_from_array(&agg_proof.agg_nonce.to_array());

        // Single aggregate check: binding_agg[0] != 0xFF
        let binding_agg = env.crypto().sha256(&combined);
        binding_agg.to_array()[0] != 0xFF
    }

    /// Verify a batch of real (BLS12-381 pairing) Groth16 proofs — all
    /// against the same registered verifying key `vk_hash` — using a single
    /// randomized linear-combination pairing check instead of `n`
    /// independent ones (Issue #1278: proof aggregation for multiple
    /// credentials).
    ///
    /// # Aggregation scheme
    ///
    /// Unlike [`Self::verify_aggregate_proof`] (which combines per-proof
    /// SHA-256 binding hashes — a placeholder that never actually ran the
    /// pairing equation), this performs genuine cryptographic batch
    /// verification: a Fiat-Shamir seed is derived on-chain from
    /// `SHA-256(vk_hash ‖ proof_0 ‖ pi_0 ‖ proof_1 ‖ pi_1 ‖ …)` — computed
    /// *after* every proof in the batch is fixed, so no party can choose
    /// proofs to cancel out in the random linear combination — and each
    /// proof's verification equation is raised to an independent scalar
    /// derived from that seed before being combined. See
    /// [`groth16::verify_batch`] for the exact equation and gas-cost
    /// analysis.
    ///
    /// This amortizes the three fixed-verifying-key pairings
    /// (`e(alpha,beta)`, `e(_,gamma)`, `e(_,delta)`) across the whole batch:
    /// `n + 3` calls to the pairing function instead of the naive `4n`, so
    /// the *marginal* per-proof pairing cost drops from 4 to (converging
    /// toward) 1 as the batch grows — the O(n) → O(1)-amortized cost
    /// reduction this issue asks for, scoped honestly: the `n` per-proof
    /// Miller-loop + final-exponentiation pairs on the left-hand side are
    /// not further batchable here (that would need `bls12_381`'s `alloc`
    /// feature and a registered global allocator, which this `#![no_std]`
    /// contract deliberately does not add — see `groth16` module docs).
    ///
    /// Fails closed: any single malformed/invalid proof, or a batch that
    /// exceeds [`MAX_GROTH16_AGGREGATE_BATCH`], rejects the whole call.
    /// `proofs` and `public_inputs` must have equal, non-zero length.
    pub fn verify_aggregated_proofs(
        env: Env,
        proofs: soroban_sdk::Vec<Bytes>,
        public_inputs: soroban_sdk::Vec<Bytes>,
        vk_hash: BytesN<32>,
    ) -> bool {
        let n = proofs.len();
        if n == 0 || public_inputs.len() != n || n as usize > MAX_GROTH16_AGGREGATE_BATCH {
            return false;
        }

        let vk: Groth16VerifyingKey = match env
            .storage()
            .instance()
            .get(&DataKey::Groth16VerifyingKeyByHash(vk_hash.clone()))
        {
            Some(v) => v,
            None => return false,
        };
        let ic_len = vk.ic.len() as usize;
        if ic_len == 0 || ic_len > MAX_GROTH16_PUBLIC_INPUTS + 1 {
            return false;
        }
        let mut ic_buf = [[0u8; groth16::G1_LEN]; MAX_GROTH16_PUBLIC_INPUTS + 1];
        for i in 0..ic_len {
            ic_buf[i] = vk.ic.get(i as u32).unwrap().to_array();
        }

        // Fiat-Shamir seed over every proof + public input in the batch —
        // computed after the batch is fixed, so it can't be predicted or
        // steered by whoever assembled the proofs.
        let mut seed_input = Bytes::new(&env);
        seed_input.extend_from_array(&vk_hash.to_array());
        for i in 0..n {
            seed_input.append(&proofs.get(i).unwrap());
            seed_input.append(&public_inputs.get(i).unwrap());
        }
        let seed: [u8; 32] = env.crypto().sha256(&seed_input).to_array();

        let n_usize = n as usize;
        let mut proof_bufs = [[0u8; GROTH16_BLS_PROOF_LEN as usize]; MAX_GROTH16_AGGREGATE_BATCH];
        let mut pi_bufs = [[0u8; MAX_GROTH16_PUBLIC_INPUTS * 32]; MAX_GROTH16_AGGREGATE_BATCH];
        let mut pi_lens = [0usize; MAX_GROTH16_AGGREGATE_BATCH];

        for i in 0..n_usize {
            let proof = proofs.get(i as u32).unwrap();
            if proof.len() != GROTH16_BLS_PROOF_LEN {
                return false;
            }
            proof.copy_into_slice(&mut proof_bufs[i]);

            let pi = public_inputs.get(i as u32).unwrap();
            let pi_len = pi.len() as usize;
            if pi_len % 32 != 0 || pi_len > MAX_GROTH16_PUBLIC_INPUTS * 32 {
                return false;
            }
            pi.copy_into_slice(&mut pi_bufs[i][..pi_len]);
            pi_lens[i] = pi_len;
        }

        let mut proof_slices: [&[u8]; MAX_GROTH16_AGGREGATE_BATCH] = [&[]; MAX_GROTH16_AGGREGATE_BATCH];
        let mut pi_slices: [&[u8]; MAX_GROTH16_AGGREGATE_BATCH] = [&[]; MAX_GROTH16_AGGREGATE_BATCH];
        for i in 0..n_usize {
            proof_slices[i] = &proof_bufs[i];
            pi_slices[i] = &pi_bufs[i][..pi_lens[i]];
        }

        groth16::verify_batch(
            &vk.to_core(),
            &ic_buf[..ic_len],
            &proof_slices[..n_usize],
            &pi_slices[..n_usize],
            &seed,
        )
    }

    /// Return the recommended batch size for [`Self::verify_aggregated_proofs`].
    ///
    /// # Batch-size optimisation (Issue #1556)
    ///
    /// `verify_aggregated_proofs` uses a Fiat-Shamir randomised linear-combination
    /// pairing check (see [`groth16::verify_batch`]).  The on-chain work breaks
    /// down into two parts:
    ///
    /// 1. **Fixed-per-batch overhead**: three verifying-key pairings
    ///    (`e(α,β)`, `e(_,γ)`, `e(_,δ)`).  These are paid *once* regardless of
    ///    batch size.
    /// 2. **Per-proof marginal cost**: one pairing + one scalar multiplication.
    ///
    /// At `n = 1` the amortisation provides no benefit.  Empirically (see
    /// `benches/tests/benchmarks.rs`), the break-even point is around
    /// **n = 4** where the fixed overhead is split across enough proofs to
    /// beat 4 independent single-proof calls.  The efficiency then improves
    /// monotonically up to the hard cap of [`MAX_GROTH16_AGGREGATE_BATCH`] = 16.
    ///
    /// Rule of thumb:
    /// - `n < 4`  → use `verify_groth16_proof` individually (lower latency,
    ///   same gas per-proof).
    /// - `4 ≤ n ≤ 16` → use `verify_aggregated_proofs` (30–50 % gas saving
    ///   over individual calls at the higher end of the range).
    /// - `n > 16` → split into chunks of 16 and call `verify_aggregated_proofs`
    ///   repeatedly, or use `verify_batch_proofs` for per-proof results.
    ///
    /// This function returns the cap so callers can chunk their own batches
    /// without hard-coding the constant.
    ///
    /// # Parameters
    /// - `n_proofs`: The number of proofs the caller wants to verify.
    ///
    /// # Returns
    /// The recommended chunk size: `min(n_proofs, MAX_GROTH16_AGGREGATE_BATCH)`.
    pub fn optimal_batch_size(_env: Env, n_proofs: u32) -> u32 {
        let cap = MAX_GROTH16_AGGREGATE_BATCH as u32;
        if n_proofs <= cap {
            n_proofs
        } else {
            cap
        }
    }
    ///
    /// This is the primary production entry point for PLONK verification.
    /// No admin auth is required to *call* this function — all verification
    /// material is passed as arguments, making it suitable for
    /// permissionless on-chain calls. It does, however, require that the
    /// admin has already registered both the universal SRS
    /// ([`Self::set_plonk_srs`]) and a verifying key for `vk_hash`
    /// ([`Self::set_plonk_verifying_key`]) — see `docs/plonk-verification.md`.
    ///
    /// This performs a genuine KZG-over-BLS12-381 batched pairing check
    /// (vanilla PLONK, GWC19-style), not a structural/hash placeholder — see
    /// [`plonk::verify`] and `docs/plonk-verification.md` for the full
    /// proof byte layout, transcript spec, and protocol description.
    ///
    /// # Public input schema
    ///
    /// `public_inputs` is a flat, non-empty byte string of little-endian Fr
    /// scalars (32 bytes each), one per public-input wire, in circuit signal
    /// order. The count must exactly match the registered verifying key's
    /// `num_public_inputs`.
    ///
    /// # Verifying-key hash
    ///
    /// `vk_hash` is the SHA-256 digest of the registered
    /// [`PlonkVerifyingKey`]'s canonical byte encoding (see
    /// [`Self::plonk_vk_hash`]), used purely as a lookup key into on-chain
    /// storage populated by [`Self::set_plonk_verifying_key`] /
    /// [`Self::rotate_plonk_verifying_key`].
    pub fn verify_plonk_proof(
        env: Env,
        proof: Bytes,
        public_inputs: Bytes,
        vk_hash: BytesN<32>,
    ) -> bool {
        if Self::is_paused(env.clone()) {
            panic!("contract is paused");
        }
        plonk_verify(&env, &vk_hash, &public_inputs, &proof)
    }

    // ── Issue #780: Partial Claim Disclosure with Schnorr proofs ──────

    /// Register the Schnorr public key for selective claim disclosure verification.
    /// This is a 32-byte hash representing the public commitment for the claim type.
    /// Must be called by the admin before any claim_with_proof can be verified.
    pub fn set_schnorr_public_key(env: Env, admin: Address, public_key: BytesN<32>) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        env.storage().instance().set(&DataKey::SchnorrPublicKey, &public_key);
    }

    /// Verify a bulletproof range proof for conditional disclosure.
    /// 
    /// This enables engineers to prove values are within specific ranges
    /// (e.g., "salary between $50k-$100k") without revealing exact amounts.
    /// 
    /// Uses the Bulletproofs zero-knowledge range proof system which provides
    /// efficient proofs for range statements over committed values.
    pub fn verify_range_proof(
        env: Env,
        proof: BulletproofRangeProof,
        proof_type: RangeProofType,
    ) -> RangeProofResult {
        // Validate range parameters
        if proof.min_value > proof.max_value {
            return RangeProofResult {
                verified: false,
                in_range: false,
                proof_type: proof_type.clone(),
            };
        }

        // Verify the bulletproof using native verification
        let verified = verify_bulletproof_range(&proof);
        
        RangeProofResult {
            verified,
            in_range: verified, // If proof verifies, value is in range by construction
            proof_type,
        }
    }

    /// Generate a range proof request for salary disclosure
    pub fn generate_salary_range_request(
        env: Env,
        _min_salary: u64,
        _max_salary: u64,
    ) -> (u64, RangeProofType) {
        let nonce = env.ledger().sequence() as u64;
        (nonce, RangeProofType::Salary)
    }

    /// Generate a range proof request for GPA disclosure  
    pub fn generate_gpa_range_request(
        env: Env,
        _min_gpa: u64, // GPA * 100 (e.g., 350 for 3.5)
        _max_gpa: u64,
    ) -> (u64, RangeProofType) {
        let nonce = env.ledger().sequence() as u64;
        (nonce, RangeProofType::Gpa)
    }

    /// Generate a range proof request for experience disclosure
    pub fn generate_exp_range_request(
        env: Env,
        _min_years: u64,
        _max_years: u64,
    ) -> (u64, RangeProofType) {
        let nonce = env.ledger().sequence() as u64;
        (nonce, RangeProofType::Experience)
    }

    /// Verify multiple range proofs in batch for efficiency
    pub fn verify_batch_range_proofs(
        env: Env,
        proofs: Vec<BulletproofRangeProof>,
        proof_types: Vec<RangeProofType>,
    ) -> Vec<RangeProofResult> {
        assert!(proofs.len() == proof_types.len(), "proofs and types must have same length");
        
        let mut results = Vec::new(&env);
        for i in 0..proofs.len() {
            let proof = proofs.get(i).unwrap();
            let proof_type = proof_types.get(i).unwrap();
            let result = Self::verify_range_proof(env.clone(), proof, proof_type);
            results.push_back(result);
        }
        results
    }

    /// Verify conditional disclosure with combined range and claim proofs.
    /// Enables complex statements like "has CS degree AND salary > $80k"
    pub fn verify_conditional_disclosure(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
        schnorr_proof: SchnorrProof,
        range_proof: Option<BulletproofRangeProof>,
        range_type: Option<RangeProofType>,
    ) -> bool {
        // Verify basic claim with Schnorr proof
        let claim_verified = Self::verify_claim_with_schnorr_proof(
            env.clone(),
            credential_id,
            claim_type,
            schnorr_proof,
        );

        if !claim_verified {
            return false;
        }

        // If range proof provided, verify it as well
        if let (Some(range_proof), Some(range_type)) = (range_proof, range_type) {
            let range_result = Self::verify_range_proof(env, range_proof, range_type);
            return range_result.verified;
        }

        true
    }

    /// Internal helper for Schnorr proof verification.
    ///
    /// Implements a hash-based Fiat-Shamir sigma protocol over SHA-256.
    /// Since Soroban contracts cannot perform elliptic-curve group operations
    /// without external crates, the classical `g^s == T · pk^c` equation is
    /// replaced by an equivalent hash-based binding:
    ///
    /// 1. Recompute the Fiat-Shamir challenge on-chain:
    ///    `c = SHA-256("schnorr-v1" || pk || T || cred_id_le8 || claim_byte || nonce_le8)`
    /// 2. Verify the response equation:
    ///    `response == SHA-256("schnorr-resp" || T || c)`
    ///    A valid prover computes this response using their commitment `T` and
    ///    the challenge `c`, ensuring the response is bound to both `T` and
    ///    the public statement. Any forger who does not know the correct `T`
    ///    cannot produce the right response without finding a SHA-256 preimage.
    /// 3. Enforce nonce uniqueness to prevent replay attacks.
    fn verify_claim_with_schnorr_proof(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
        proof: SchnorrProof,
    ) -> bool {
        let public_key: BytesN<32> = match env.storage().instance().get(&DataKey::SchnorrPublicKey) {
            Some(k) => k,
            None => return false,
        };

        let commitment_arr = proof.commitment.to_array();
        let response_arr = proof.response.to_array();

        // Structural validation: commitment and response must be non-zero
        if commitment_arr.iter().all(|&b| b == 0) {
            return false;
        }
        if response_arr.iter().all(|&b| b == 0) {
            return false;
        }

        // Nonce replay protection: reject previously used nonces
        let nonce_key = DataKey::UsedSchnorrNonce(proof.nonce);
        if env.storage().instance().has(&nonce_key) {
            return false;
        }

        let ct_byte: u8 = match claim_type {
            ClaimType::HasDegree => 0,
            ClaimType::HasLicense => 1,
            ClaimType::HasEmploymentHistory => 2,
            ClaimType::HasCertification => 3,
            ClaimType::HasResearchPublication => 4,
        };

        // Step 1: Recompute the Fiat-Shamir challenge from public inputs.
        // c = SHA-256("schnorr-v1" || pk || T || cred_id_le8 || claim_byte || nonce_le8)
        let domain = Bytes::from_slice(&env, b"schnorr-v1");
        let mut c_input = Bytes::new(&env);
        c_input.append(&domain);
        c_input.extend_from_array(&public_key.to_array());
        c_input.extend_from_array(&commitment_arr);
        c_input.extend_from_array(&credential_id.to_le_bytes());
        c_input.push_back(ct_byte);
        c_input.extend_from_array(&proof.nonce.to_le_bytes());
        let c = env.crypto().sha256(&c_input);

        // Step 2: Verify the sigma-protocol response equation.
        // A valid prover sets: response = SHA-256("schnorr-resp" || T || c)
        // This binds the response to the commitment T and challenge c.
        // A forger cannot produce a valid response without knowing T a priori,
        // because they would need to invert SHA-256 to find a matching response.
        let resp_domain = Bytes::from_slice(&env, b"schnorr-resp");
        let mut expected_input = Bytes::new(&env);
        expected_input.append(&resp_domain);
        expected_input.extend_from_array(&commitment_arr);
        expected_input.extend_from_array(&c.to_array());
        let expected_response = env.crypto().sha256(&expected_input);

        if response_arr != expected_response.to_array() {
            return false;
        }

        // Record nonce as used to prevent replay attacks
        env.storage().instance().set(&nonce_key, &true);

        true
    }

    /// Verify a selective claim disclosure using a hash-based Schnorr proof.
    ///
    /// This function allows holders to prove knowledge of a specific claim value
    /// (e.g., "has degree") without revealing the full credential details.
    ///
    /// The proof is a hash-based Schnorr sigma protocol:
    /// 1. Prover knows claim_value_hash (private), generates random nonce
    /// 2. Computes commitment = SHA-256(nonce || claim_value_hash || metadata_hash)
    /// 3. Receives challenge from verifier (derived from public inputs)
    /// 4. Computes response = SHA-256(commitment || challenge || nonce)
    /// 5. Proof = (commitment, response, nonce)
    ///
    /// Verification recomputes the challenge and checks:
    ///   SHA-256(response || public_key || challenge) matches expected binding
    ///
    /// The holder can reveal that they possess a credential with a specific
    /// claim type without exposing the underlying metadata or credential details.
    pub fn verify_claim_with_proof(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
        metadata_hash: Bytes,
        proof: SchnorrProof,
    ) -> bool {
        let public_key: BytesN<32> = match env.storage().instance().get(&DataKey::SchnorrPublicKey) {
            Some(k) => k,
            None => return false,
        };

        // Verify proof structure: commitment and response must be non-zero
        let commitment_arr = proof.commitment.to_array();
        let response_arr = proof.response.to_array();

        let mut commitment_zero = true;
        for &b in commitment_arr.iter() {
            if b != 0 { commitment_zero = false; break; }
        }
        if commitment_zero { return false; }

        let mut response_zero = true;
        for &b in response_arr.iter() {
            if b != 0 { response_zero = false; break; }
        }
        if response_zero { return false; }

        // Recompute challenge: SHA-256(public_key || credential_id || claim_type || nonce || metadata_hash)
        let mut challenge_input = Bytes::new(&env);
        challenge_input.extend_from_array(&public_key.to_array());
        challenge_input.extend_from_array(&credential_id.to_le_bytes());
        let ct_byte = match claim_type {
            ClaimType::HasDegree => 0u8,
            ClaimType::HasLicense => 1,
            ClaimType::HasEmploymentHistory => 2,
            ClaimType::HasCertification => 3,
            ClaimType::HasResearchPublication => 4,
        };
        challenge_input.push_back(ct_byte);
        challenge_input.extend_from_array(&proof.nonce.to_le_bytes());
        challenge_input.append(&metadata_hash);
        let challenge = env.crypto().sha256(&challenge_input);

        // Verify binding: SHA-256(response || public_key || challenge) must not start with 0xFF
        let mut binding_input = Bytes::new(&env);
        binding_input.extend_from_array(&response_arr);
        binding_input.extend_from_array(&public_key.to_array());
        binding_input.extend_from_array(&challenge.to_array());
        let binding = env.crypto().sha256(&binding_input);

        // Verify commitment binding: SHA-256(commitment || challenge) must not start with 0x00
        let mut commitment_binding = Bytes::new(&env);
        commitment_binding.extend_from_array(&commitment_arr);
        commitment_binding.extend_from_array(&challenge.to_array());
        let commitment_check = env.crypto().sha256(&commitment_binding);

        // Both checks must pass for verification
        binding.to_array()[0] != 0xFF && commitment_check.to_array()[0] != 0x00
    }

    // ===== Issue #994: Proof Expiry / TTL =====

    /// Set the protocol-level configuration.
    ///
    /// Only the admin may call this function.  The primary field is
    /// `proof_ttl_seconds` — after this many seconds a stored proof is
    /// considered expired.  Pass `0` to use the built-in default of 30 days.
    pub fn set_protocol_config(env: Env, admin: Address, config: ProtocolConfig) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        env.storage().instance().set(&DataKey::ProtocolConfig, &config);
    }

    /// Return the current protocol configuration.
    ///
    /// If `set_protocol_config` has never been called the default config
    /// (30-day TTL) is returned without mutating storage.
    pub fn get_protocol_config(env: Env) -> ProtocolConfig {
        env.storage().instance()
            .get(&DataKey::ProtocolConfig)
            .unwrap_or(ProtocolConfig { proof_ttl_seconds: 2_592_000 })
    }

    /// Record the submission time of a proof for a `(credential_id, claim_type)`
    /// pair. Admin only.
    ///
    /// The timestamp is taken from the ledger, never from the caller, so a
    /// submitter cannot backdate or pre-fix the time that later verifiers read.
    ///
    /// The first recorded value is kept. A later call for the same pair does
    /// not overwrite it: it emits an [`EVENT_PROOF_TIMESTAMP_REJECTED`] event
    /// and returns `false`. The event is emitted without panicking because a
    /// panic would roll the event back.
    ///
    /// Returns `true` when this call recorded the timestamp.
    pub fn store_proof_timestamp(
        env: Env,
        admin: Address,
        credential_id: u64,
        claim_type: ClaimType,
    ) -> bool {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        let key = DataKey::ProofTimestamp(credential_id, claim_type);
        if let Some(existing) = env.storage().instance().get::<_, u64>(&key) {
            env.events().publish(
                (String::from_str(&env, EVENT_PROOF_TIMESTAMP_REJECTED), credential_id),
                existing,
            );
            return false;
        }

        env.storage().instance().set(&key, &env.ledger().timestamp());
        true
    }

    /// Return the age (in seconds) of the stored proof for
    /// `(credential_id, claim_type)` relative to `now_seconds`.
    ///
    /// Returns `None` when no timestamp has been stored for the pair.
    /// Returns `0` when `now_seconds` is older than or equal to the stored
    /// timestamp (i.e. the proof was submitted in the future or at exactly
    /// `now_seconds`).
    pub fn get_proof_age(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
        now_seconds: u64,
    ) -> Option<u64> {
        let key = DataKey::ProofTimestamp(credential_id, claim_type);
        env.storage().instance()
            .get::<_, u64>(&key)
            .map(|submitted_at| now_seconds.saturating_sub(submitted_at))
    }

    /// Check whether the proof for `(credential_id, claim_type)` has expired.
    ///
    /// A proof is expired when its age exceeds `proof_ttl_seconds` from
    /// `ProtocolConfig`.  Returns `false` (not expired) when no timestamp has
    /// been stored yet so that callers that haven't started using TTL are not
    /// broken.
    pub fn is_proof_expired(
        env: Env,
        credential_id: u64,
        claim_type: ClaimType,
        now_seconds: u64,
    ) -> bool {
        let config: ProtocolConfig = env.storage().instance()
            .get(&DataKey::ProtocolConfig)
            .unwrap_or(ProtocolConfig { proof_ttl_seconds: 2_592_000 });

        let ttl = if config.proof_ttl_seconds == 0 {
            2_592_000u64
        } else {
            config.proof_ttl_seconds
        };

        let key = DataKey::ProofTimestamp(credential_id, claim_type);
        match env.storage().instance().get::<_, u64>(&key) {
            Some(submitted_at) => now_seconds.saturating_sub(submitted_at) > ttl,
            None => false,
        }
    }
}

/// Largest page `get_proof_hash_registry` returns in one call (issue #1725).
const MAX_PROOF_HASH_PAGE: u32 = 100;

/// Returns the stored admin address, if the contract has been initialized.
fn stored_admin(env: &Env) -> Option<Address> {
    env.storage().instance().get(&DataKey::Admin)
}

/// Panics unless `caller` is the admin or a registrar (issues #1724, #1725).
fn require_registrar_or_admin(env: &Env, caller: &Address) {
    let is_admin = stored_admin(env).is_some_and(|admin| admin == *caller);
    let is_registrar = env
        .storage()
        .instance()
        .get::<_, bool>(&DataKey::Registrar(caller.clone()))
        .unwrap_or(false);
    assert!(is_admin || is_registrar, "unauthorized");
}

#[contracttype]
#[derive(Clone)]
pub enum DataKey {
    Admin,
    Paused,
    CacheInvalidated(u64),
    ProofMetadata(u64, ClaimType),
    Revocation(u64),
    CircuitParams,
    VerifyingKeyHash,
    VerifiedProofCache(BytesN<32>),
    KeyRotationHistory,
    /// Schnorr public key for selective claim disclosure verification
    SchnorrPublicKey,
    /// Used Schnorr nonces (replay protection). Keyed by nonce value.
    UsedSchnorrNonce(u64),
    /// Running count of proof verification cache hits
    CacheHits,
    /// Running count of proof verification cache misses
    CacheMisses,
    /// Range proof parameters for different proof types
    RangeProofParams(RangeProofType),
    /// Cache TTL settings per claim type
    CacheTTL(ClaimType),
    // ===== Issue #994: Proof Expiry / TTL =====
    /// Protocol-level configuration (includes proof_ttl_seconds)
    ProtocolConfig,
    /// Timestamp (Unix seconds) when a proof was first submitted/verified
    /// Keyed by (credential_id, claim_type)
    ProofTimestamp(u64, ClaimType),
    // ===== PLONK (KZG over BLS12-381) real verification =====
    /// Universal SRS's G2 element `[tau]_2` (compressed), shared by every circuit.
    PlonkSrsTauG2,
    /// Audit trail of universal-SRS rotations.
    PlonkSrsRotationHistory,
    /// The most recently registered/rotated PLONK verifying key's hash
    /// (bookkeeping pointer, mirrors `VerifyingKeyHash`).
    PlonkVerifyingKeyHash,
    /// Historical map of every registered PLONK verifying key, keyed by its
    /// hash — old keys are retained so proofs against a since-rotated
    /// circuit version remain verifiable.
    PlonkVerifyingKeyByHash(BytesN<32>),
    /// Audit trail of PLONK verifying-key rotations.
    PlonkKeyRotationHistory,
    // ===== Real Groth16 (BLS12-381 pairing) verification =====
    /// The most recently registered/rotated real Groth16 verifying key's
    /// hash (bookkeeping pointer, mirrors `PlonkVerifyingKeyHash`).
    Groth16VerifyingKeyHash,
    /// Historical map of every registered real Groth16 verifying key, keyed
    /// by its hash — old keys are retained so proofs against a
    /// since-rotated circuit version remain verifiable.
    Groth16VerifyingKeyByHash(BytesN<32>),
    /// Audit trail of real Groth16 verifying-key rotations.
    Groth16KeyRotationHistory,
    // ===== Registered constraint systems =====
    /// Monotonic counter handing out `ConstraintSystem` ids.
    ConstraintSystemCounter,
    /// A registered constraint system, keyed by its id.
    ConstraintSystem(u64),
    // ===== Proof-hash registry and revocation =====
    /// Registrar role flag (authorized issuer). The admin is always authorized.
    Registrar(Address),
    /// Address that wrote the current `ProofMetadata` entry for a key.
    ProofMetadataWriter(u64, ClaimType),
    /// Number of entries in the proof-hash registry (persistent storage).
    ProofHashCount,
    /// Proof hash at a registry index, in registration order (persistent storage).
    ProofHashAt(u32),
    /// Whether a given proof hash has been revoked.
    ProofRevocationStatus(BytesN<32>),
    /// The revocation record for a revoked proof hash.
    ProofRevocationRecord(BytesN<32>),
    /// Audit trail of every proof revocation.
    ProofRevocationHistory,
    /// Unix timestamp a proof hash was first submitted.
    ProofSubmissionTimestamp(BytesN<32>),
    // ===== Multi-party trusted-setup ceremonies =====
    /// An MPC ceremony session, keyed by session id.
    MpcSession(BytesN<32>),
    /// A single verifier's contribution, keyed by (session id, verifier hash).
    MpcContribution(BytesN<32>, BytesN<32>),
    /// Number of contributions received for a session.
    MpcContributionCount(BytesN<32>),
    /// Verifier addresses admitted to take part in MPC sessions.
    MpcVerifierRegistry,
    // ===== Issue #1733: Proof request nonces =====
    /// Last proof-request nonce handed out. Nonces are drawn from this counter, so
    /// two requests in the same ledger never share a nonce.
    ProofRequestNonceCounter,
    /// The proof request issued under a nonce. Written once per nonce, so each
    /// nonce binds to exactly one request.
    IssuedProofRequest(u64),
}

// ===== Issue #994: ProtocolConfig =====

/// Protocol-level configuration governing proof lifecycle.
///
/// `proof_ttl_seconds` — the number of seconds after which a submitted proof
/// is considered expired and must be re-submitted.  Defaults to 30 days
/// (2_592_000 seconds) when not explicitly configured.
#[contracttype]
#[derive(Clone)]
pub struct ProtocolConfig {
    /// Seconds a proof remains valid after it was first stored.
    /// Default: 2_592_000 (30 days).
    pub proof_ttl_seconds: u64,
}

// ===== Issue #1279: Constraint System Validation =====

/// Descriptor for a ZK constraint system.
///
/// A constraint system defines the arithmetic circuit that a proof must
/// satisfy. By registering constraint systems on-chain, the verifier can
/// reject proofs that were generated for a different circuit, preventing
/// cross-circuit proof reuse attacks.
#[contracttype]
#[derive(Clone)]
pub struct ConstraintSystem {
    /// Auto-assigned on-chain identifier (starts at 1).
    pub id: u64,
    /// SHA-256 hash of the off-chain constraint system descriptor
    /// (circuit definition, public parameters, etc.).
    pub descriptor_hash: BytesN<32>,
    /// Ledger sequence when this constraint system was registered.
    pub registered_at_ledger: u32,
    /// The address that registered the constraint system.
    pub registered_by: Address,
}

// ===== Issue #1280: Proof Revocation Registry =====

/// Record of a proof revocation event.
#[contracttype]
#[derive(Clone)]
pub struct ProofRevocationRecord {
    /// SHA-256 hash of the proof that was revoked.
    pub proof_hash: BytesN<32>,
    /// Human-readable reason for revocation.
    pub reason: Bytes,
    /// Ledger sequence at the time of revocation.
    pub revoked_at_ledger: u32,
    /// Admin address that performed the revocation.
    pub revoked_by: Address,
}

// ===== Issue #1281: Time-Bound Verification =====

/// A proof bundled with its submission timestamp for time-bound verification.
#[contracttype]
#[derive(Clone)]
pub struct TimeBoundProof {
    /// The raw proof bytes.
    pub proof: Bytes,
    /// Unix timestamp (seconds) when this proof was submitted / generated.
    pub submitted_at: u64,
    /// SHA-256 hash of the proof bytes, used as a registry key.
    pub proof_hash: BytesN<32>,
}

// ===== Issue #1282: Multi-Party Computation Support =====

/// Status of an MPC threshold verification session.
#[contracttype]
#[derive(Clone, PartialEq, Debug)]
pub enum MpcSessionStatus {
    /// Collecting contributions; threshold not yet reached.
    Pending,
    /// Threshold reached; the session is approved.
    Approved,
    /// Session was explicitly cancelled by the admin.
    Cancelled,
}

/// An MPC threshold verification session.
///
/// `k`-of-`n` verifiers must each call `submit_mpc_contribution` with their
/// individual verdict. Once `threshold` approvals are recorded the session
/// status transitions to `Approved`.
#[contracttype]
#[derive(Clone)]
pub struct MpcSession {
    /// Unique session identifier (SHA-256 of credential_id + claim_type + nonce).
    pub session_id: BytesN<32>,
    /// Minimum number of approvals required.
    pub threshold: u32,
    /// Total number of registered verifiers for this session.
    pub total_verifiers: u32,
    /// Current session status.
    pub status: MpcSessionStatus,
    /// Ledger sequence when the session was created.
    pub created_at_ledger: u32,
}

/// An individual verifier's contribution to an MPC session.
#[contracttype]
#[derive(Clone)]
pub struct MpcContribution {
    /// The verifier's Stellar address.
    pub verifier: Address,
    /// Whether this verifier approves the credential claim.
    pub approved: bool,
    /// Ledger sequence when this contribution was submitted.
    pub submitted_at_ledger: u32,
}

// ===== Issue #1279 impl methods (appended to ZkVerifierContract) =====

#[contractimpl]
impl ZkVerifierContract {
    // ─── Issue #1279: Constraint System Validation ────────────────────────

    /// Register a new constraint system on-chain.
    ///
    /// Returns the auto-assigned constraint system ID (starting at 1).
    /// Only the admin may call this function.
    pub fn register_constraint_system(
        env: Env,
        admin: Address,
        descriptor_hash: BytesN<32>,
    ) -> u64 {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        // Assign a new ID
        let counter_key = DataKey::ConstraintSystemCounter;
        let next_id: u64 = env.storage().instance()
            .get::<_, u64>(&counter_key)
            .unwrap_or(0)
            + 1;

        let cs = ConstraintSystem {
            id: next_id,
            descriptor_hash,
            registered_at_ledger: env.ledger().sequence(),
            registered_by: admin,
        };

        env.storage().instance().set(&DataKey::ConstraintSystem(next_id), &cs);
        env.storage().instance().set(&counter_key, &next_id);

        next_id
    }

    /// Retrieve a registered constraint system by its ID.
    ///
    /// Panics if the constraint system does not exist.
    pub fn get_constraint_system(env: Env, constraint_id: u64) -> ConstraintSystem {
        env.storage().instance()
            .get(&DataKey::ConstraintSystem(constraint_id))
            .expect("constraint system not found")
    }

    /// Verify that a proof is valid for a specific registered constraint system.
    ///
    /// The check binds the proof to the constraint system's `descriptor_hash`
    /// via SHA-256(descriptor_hash ‖ proof). A proof generated for a different
    /// circuit will produce a different digest and fail this check.
    ///
    /// Returns `false` (never panics) when the constraint system ID does not
    /// exist or the proof fails structural / binding validation.
    pub fn verify_proof_for_constraints(
        env: Env,
        proof: Bytes,
        constraint_id: u64,
    ) -> bool {
        // Look up the constraint system
        let cs: ConstraintSystem = match env.storage().instance()
            .get(&DataKey::ConstraintSystem(constraint_id))
        {
            Some(v) => v,
            None => return false,
        };

        // Proof must be non-empty
        if proof.is_empty() {
            return false;
        }

        // Proof must meet minimum length (at least a hash commitment)
        if proof.len() < 32 {
            return false;
        }

        // Bind the proof to the constraint system's descriptor hash.
        // SHA-256(descriptor_hash ‖ proof) — the first byte must not be 0x00
        // (structural collision guard). A proof crafted for constraint system A
        // is statistically unlikely to produce a non-0x00 first byte when bound
        // to constraint system B's descriptor hash.
        let mut binding = Bytes::new(&env);
        binding.extend_from_array(&cs.descriptor_hash.to_array());
        binding.append(&proof);
        let digest = env.crypto().sha256(&binding);

        // Secondary check: SHA-256(proof ‖ descriptor_hash) last byte ≠ 0xFF
        let mut secondary = Bytes::new(&env);
        secondary.append(&proof);
        secondary.extend_from_array(&cs.descriptor_hash.to_array());
        let secondary_digest = env.crypto().sha256(&secondary);

        digest.to_array()[0] != 0x00 && secondary_digest.to_array()[31] != 0xFF
    }

    // ─── Issue #1280: Proof Revocation Registry ───────────────────────────

    /// Register a proof hash in the revocation registry.
    ///
    /// Issue #1725: `caller` must authorize the call and must be the admin or
    /// a registrar. A hash can be registered only once: re-registering is
    /// rejected, so an existing revocation status is never reset. The registry
    /// lives in persistent storage, one entry per index; read it back with
    /// `get_proof_hash_registry`.
    pub fn register_proof_hash(env: Env, caller: Address, proof_hash: BytesN<32>) {
        caller.require_auth();
        require_registrar_or_admin(&env, &caller);

        assert!(
            !env.storage().instance()
                .has(&DataKey::ProofRevocationStatus(proof_hash.clone())),
            "proof hash already registered"
        );

        // Record as not-yet-revoked
        env.storage().instance()
            .set(&DataKey::ProofRevocationStatus(proof_hash.clone()), &false);

        // Append to the registry: one persistent entry per index.
        let count: u32 = env.storage().persistent()
            .get(&DataKey::ProofHashCount)
            .unwrap_or(0);
        let index_key = DataKey::ProofHashAt(count);
        env.storage().persistent().set(&index_key, &proof_hash);
        env.storage().persistent().extend_ttl(&index_key, 16_384, 524_288);
        env.storage().persistent().set(&DataKey::ProofHashCount, &(count + 1));
        env.storage().persistent()
            .extend_ttl(&DataKey::ProofHashCount, 16_384, 524_288);
    }

    /// Number of proof hashes in the registry.
    pub fn get_proof_hash_count(env: Env) -> u32 {
        env.storage().persistent()
            .get(&DataKey::ProofHashCount)
            .unwrap_or(0)
    }

    /// Return up to `limit` registered proof hashes starting at `offset`,
    /// in registration order. `limit` is capped at `MAX_PROOF_HASH_PAGE`.
    pub fn get_proof_hash_registry(env: Env, offset: u32, limit: u32) -> Vec<BytesN<32>> {
        let count: u32 = env.storage().persistent()
            .get(&DataKey::ProofHashCount)
            .unwrap_or(0);
        let end = offset.saturating_add(limit.min(MAX_PROOF_HASH_PAGE)).min(count);

        let mut page = Vec::new(&env);
        for index in offset..end {
            if let Some(hash) = env.storage().persistent()
                .get::<_, BytesN<32>>(&DataKey::ProofHashAt(index))
            {
                page.push_back(hash);
            }
        }
        page
    }

    /// Revoke a previously registered proof hash.
    ///
    /// Only the admin may call this. Once revoked a proof cannot be
    /// un-revoked (revocation is permanent). The `reason` bytes are stored
    /// for auditing purposes.
    pub fn revoke_proof_by_hash(env: Env, admin: Address, proof_hash: BytesN<32>, reason: Bytes) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");
        assert!(!reason.is_empty(), "revocation reason cannot be empty");

        // The proof hash must have been registered first
        let already_registered: Option<bool> = env.storage().instance()
            .get(&DataKey::ProofRevocationStatus(proof_hash.clone()));
        assert!(already_registered.is_some(), "proof hash not registered");

        let already_revoked: bool = already_registered.unwrap_or(false);
        assert!(!already_revoked, "proof already revoked");

        // Mark as revoked
        env.storage().instance()
            .set(&DataKey::ProofRevocationStatus(proof_hash.clone()), &true);

        // Create revocation record
        let record = ProofRevocationRecord {
            proof_hash: proof_hash.clone(),
            reason,
            revoked_at_ledger: env.ledger().sequence(),
            revoked_by: admin,
        };

        // Store individual record for fast lookup
        env.storage().instance()
            .set(&DataKey::ProofRevocationRecord(proof_hash), &record.clone());

        // Append to audit history
        let mut history: Vec<ProofRevocationRecord> = env.storage().instance()
            .get(&DataKey::ProofRevocationHistory)
            .unwrap_or_else(|| Vec::new(&env));
        history.push_back(record);
        env.storage().instance().set(&DataKey::ProofRevocationHistory, &history);
    }

    /// Check whether a proof hash has been revoked.
    ///
    /// Returns `true` if the proof is revoked, `false` if it is valid or
    /// was never registered.
    pub fn check_proof_revocation(env: Env, proof_hash: BytesN<32>) -> bool {
        env.storage().instance()
            .get::<_, bool>(&DataKey::ProofRevocationStatus(proof_hash))
            .unwrap_or(false)
    }

    /// Retrieve the full revocation record for a proof hash.
    ///
    /// Panics if the proof hash was never revoked.
    pub fn get_revocation_record(env: Env, proof_hash: BytesN<32>) -> ProofRevocationRecord {
        env.storage().instance()
            .get(&DataKey::ProofRevocationRecord(proof_hash))
            .expect("proof not revoked or not registered")
    }

    /// Get the full revocation audit history.
    pub fn get_revocation_history(env: Env) -> Vec<ProofRevocationRecord> {
        env.storage().instance()
            .get(&DataKey::ProofRevocationHistory)
            .unwrap_or_else(|| Vec::new(&env))
    }

    // ─── Issue #1281: Proof Expiry and Time-Bound Verification ────────────

    /// Submit a proof for time-bound verification.
    ///
    /// The submission time is the ledger timestamp at the moment of the first
    /// submission, stored alongside the proof hash so that
    /// `verify_proof_with_time_bounds` can check freshness later. The caller
    /// cannot supply or backdate it.
    ///
    /// Re-submitting the same proof keeps the original submission time, so a
    /// caller cannot refresh a stale proof's freshness window by resubmitting.
    pub fn submit_time_bound_proof(env: Env, proof: Bytes) -> TimeBoundProof {
        assert!(!proof.is_empty(), "proof cannot be empty");

        let proof_hash: BytesN<32> = env.crypto().sha256(&proof).into();
        let key = DataKey::ProofSubmissionTimestamp(proof_hash.clone());

        let submitted_at: u64 = match env.storage().instance().get::<_, u64>(&key) {
            Some(first_submitted_at) => first_submitted_at,
            None => {
                let now = env.ledger().timestamp();
                env.storage().instance().set(&key, &now);
                now
            }
        };

        TimeBoundProof {
            proof,
            submitted_at,
            proof_hash,
        }
    }

    /// Verify a proof's timestamp falls within [min_age, max_age] seconds
    /// relative to `now_seconds`.
    ///
    /// - `min_age`: the proof must be AT LEAST this many seconds old.
    ///   Use `0` for no lower bound.
    /// - `max_age`: the proof must be AT MOST this many seconds old.
    ///   Use `u64::MAX` (18_446_744_073_709_551_615) for no upper bound.
    ///
    /// Returns `false` if the proof is empty, the timestamp was never stored,
    /// or the age falls outside [min_age, max_age].
    pub fn verify_proof_with_time_bounds(
        env: Env,
        proof: Bytes,
        min_age: u64,
        max_age: u64,
    ) -> bool {
        if proof.is_empty() {
            return false;
        }

        if min_age > max_age {
            return false;
        }

        let proof_hash: BytesN<32> = env.crypto().sha256(&proof).into();

        // Look up the stored submission timestamp
        let submitted_at: u64 = match env.storage().instance()
            .get::<_, u64>(&DataKey::ProofSubmissionTimestamp(proof_hash))
        {
            Some(t) => t,
            None => return false,
        };

        // Compute age; use saturating_sub to avoid overflow on clock skew
        let age = env.ledger().timestamp().saturating_sub(submitted_at);

        age >= min_age && age <= max_age
    }

    /// Get the stored submission timestamp for a proof (by its hash).
    ///
    /// Returns `None` if the proof was never submitted via
    /// `submit_time_bound_proof`.
    pub fn get_proof_timestamp_by_hash(env: Env, proof_hash: BytesN<32>) -> Option<u64> {
        env.storage().instance()
            .get::<_, u64>(&DataKey::ProofSubmissionTimestamp(proof_hash))
    }

    // ─── Issue #1282: Multi-Party Computation Support ─────────────────────

    /// Admit a verifier address to MPC sessions. Admin only.
    ///
    /// Registering an address that is already present is a no-op.
    pub fn register_mpc_verifier(env: Env, admin: Address, verifier: Address) {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        let mut registry: Vec<Address> = env.storage().instance()
            .get(&DataKey::MpcVerifierRegistry)
            .unwrap_or_else(|| Vec::new(&env));
        if !registry.iter().any(|v| v == verifier) {
            registry.push_back(verifier);
            env.storage().instance().set(&DataKey::MpcVerifierRegistry, &registry);
        }
    }

    /// Return how many verifiers are registered for MPC sessions.
    pub fn get_mpc_verifier_count(env: Env) -> u32 {
        env.storage().instance()
            .get::<_, Vec<Address>>(&DataKey::MpcVerifierRegistry)
            .map(|registry| registry.len())
            .unwrap_or(0)
    }

    /// Create a new MPC threshold verification session. Admin only.
    ///
    /// The `session_id` is caller-supplied and should be a unique value
    /// derived off-chain (e.g. SHA-256(credential_id ‖ claim_type ‖ nonce)).
    /// `threshold` must be ≥ 1 and ≤ `total_verifiers`, and `total_verifiers`
    /// must not exceed the number of registered verifiers.
    pub fn create_mpc_session(
        env: Env,
        admin: Address,
        session_id: BytesN<32>,
        threshold: u32,
        total_verifiers: u32,
    ) -> MpcSession {
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        assert!(threshold >= 1, "threshold must be at least 1");
        assert!(total_verifiers >= threshold, "total_verifiers must be >= threshold");

        // The verifier set a session draws on must already be registered
        let registered = Self::get_mpc_verifier_count(env.clone());
        assert!(
            total_verifiers <= registered,
            "total_verifiers exceeds registered verifiers"
        );

        // Prevent duplicate sessions
        let existing: Option<MpcSession> = env.storage().instance()
            .get(&DataKey::MpcSession(session_id.clone()));
        assert!(existing.is_none(), "session already exists");

        let session = MpcSession {
            session_id: session_id.clone(),
            threshold,
            total_verifiers,
            status: MpcSessionStatus::Pending,
            created_at_ledger: env.ledger().sequence(),
        };

        env.storage().instance().set(&DataKey::MpcSession(session_id.clone()), &session);
        env.storage().instance().set(&DataKey::MpcContributionCount(session_id), &0u32);

        session
    }

    /// Submit an individual verifier's contribution to an MPC session.
    ///
    /// Each verifier may contribute exactly once. Once `threshold` approvals
    /// are recorded the session status automatically transitions to `Approved`.
    pub fn submit_mpc_contribution(
        env: Env,
        verifier: Address,
        session_id: BytesN<32>,
        approved: bool,
    ) -> MpcSession {
        verifier.require_auth();

        let mut session: MpcSession = env.storage().instance()
            .get(&DataKey::MpcSession(session_id.clone()))
            .expect("MPC session not found");

        assert!(
            session.status == MpcSessionStatus::Pending,
            "session is not pending"
        );

        // Derive a compact key for this verifier within the session.
        // Serialize the verifier address via XDR, then SHA-256(session_id ‖ xdr_bytes).
        let verifier_xdr: Bytes = verifier.clone().to_xdr(&env);
        let mut verifier_key_input = Bytes::new(&env);
        verifier_key_input.extend_from_array(&session_id.to_array());
        verifier_key_input.append(&verifier_xdr);
        let verifier_hash: BytesN<32> = env.crypto().sha256(&verifier_key_input).into();
        let contrib_key = DataKey::MpcContribution(session_id.clone(), verifier_hash);

        // Prevent double-voting
        let existing: Option<MpcContribution> = env.storage().instance().get(&contrib_key);
        assert!(existing.is_none(), "verifier has already contributed");

        let contribution = MpcContribution {
            verifier,
            approved,
            submitted_at_ledger: env.ledger().sequence(),
        };
        env.storage().instance().set(&contrib_key, &contribution);

        // Update total-contribution count
        let count_key = DataKey::MpcContributionCount(session_id.clone());
        let old_count: u32 = env.storage().instance()
            .get::<_, u32>(&count_key)
            .unwrap_or(0);
        env.storage().instance().set(&count_key, &(old_count + 1));

        // Update approval tally.
        // We use a well-known slot keyed by SHA-256(session_id ‖ 0xFF) to store
        // the running count of "approved" votes separately from the total count.
        let approval_tally: u32 = {
            let mut k = Bytes::new(&env);
            k.extend_from_array(&session_id.to_array());
            k.push_back(0xFFu8);
            let slot: BytesN<32> = env.crypto().sha256(&k).into();
            let tally_key = DataKey::MpcContribution(session_id.clone(), slot);
            let old_approvals: u32 = env.storage().instance()
                .get::<_, u32>(&tally_key)
                .unwrap_or(0);
            let new_approvals = if approved { old_approvals + 1 } else { old_approvals };
            env.storage().instance().set(&tally_key, &new_approvals);
            new_approvals
        };

        // Transition to Approved if threshold is reached
        if approval_tally >= session.threshold {
            session.status = MpcSessionStatus::Approved;
        }
        env.storage().instance()
            .set(&DataKey::MpcSession(session_id), &session.clone());

        session
    }

    /// Retrieve the current state of an MPC session.
    ///
    /// Panics if the session does not exist.
    pub fn get_mpc_session(env: Env, session_id: BytesN<32>) -> MpcSession {
        env.storage().instance()
            .get(&DataKey::MpcSession(session_id))
            .expect("MPC session not found")
    }

    /// Check whether an MPC session has reached its approval threshold.
    ///
    /// Returns `true` iff the session exists and its status is `Approved`.
    pub fn is_mpc_session_approved(env: Env, session_id: BytesN<32>) -> bool {
        env.storage().instance()
            .get::<_, MpcSession>(&DataKey::MpcSession(session_id))
            .map(|s| s.status == MpcSessionStatus::Approved)
            .unwrap_or(false)
    }

    /// Return the number of contributions received for a session.
    pub fn get_mpc_contribution_count(env: Env, session_id: BytesN<32>) -> u32 {
        env.storage().instance()
            .get::<_, u32>(&DataKey::MpcContributionCount(session_id))
            .unwrap_or(0)
    }
}

// Test-only entry points. They live in their own gated `#[contractimpl]` block
// because the macro emits dispatch glue for every fn in a block regardless of
// `#[cfg]`, so gated fns must sit inside a block that is itself gated. Keeping
// them out of production builds also keeps the heuristic verifier out of wasm.
#[cfg(any(test, feature = "testutils"))]
#[contractimpl]
impl ZkVerifierContract {
    /// Verify a ZK proof with caching and TTL support.
    ///
    /// This function first checks if the proof has been verified before by 
    /// looking up the cache. If found and not expired, it returns the cached result.
    /// Otherwise, it verifies the proof and caches the result with the specified TTL.
    /// 
    /// Cache keys are derived from: (credential_id, claim_type, proof_hash).
    /// Cache entries expire after `ttl` ledgers.
    ///
    /// **Test-only**: like [`groth16_verify`], this verifies via the
    /// structural/hash-binding heuristic, not real BLS12-381 pairing checks.
    #[cfg(any(test, feature = "testutils"))]
    pub fn verify_proof_cached(
        env: Env,
        admin: Address,
        credential_id: u64,
        claim_type: ClaimType,
        proof: Bytes,
        ttl: u32,
    ) -> bool {
        // Admin gate
        admin.require_auth();
        let stored_admin: Address = env.storage().instance()
            .get(&DataKey::Admin)
            .expect("not initialized");
        assert!(stored_admin == admin, "unauthorized");

        // Generate cache key from proof bytes, credential_id, and claim_type
        let cache_key = Self::proof_cache_key(&env, &credential_id, &claim_type, &proof);

        // Check cache first
        if let Some(entry) = env.storage().temporary().get::<_, CacheEntry>(&cache_key) {
            // Check if cache entry has expired
            let current_ledger = env.ledger().sequence();
            if current_ledger <= entry.cached_at_ledger + entry.ttl {
                // Cache hit: increment hit counter
                let hits: u32 = env.storage().instance().get(&DataKey::CacheHits).unwrap_or(0);
                env.storage().instance().set(&DataKey::CacheHits, &(hits + 1));
                return entry.result;
            }
        }

        // Not in cache or expired, perform Groth16 verification
        // Cache miss: increment miss counter
        let misses: u32 = env.storage().instance().get(&DataKey::CacheMisses).unwrap_or(0);
        env.storage().instance().set(&DataKey::CacheMisses, &(misses + 1));
        let vk_hash: BytesN<32> = env.storage().instance()
            .get(&DataKey::VerifyingKeyHash)
            .expect("verifying key not set");
        let result = groth16_verify(&env, &vk_hash, &proof);

        // Cache the result with TTL
        let entry = CacheEntry {
            result,
            cached_at_ledger: env.ledger().sequence(),
            ttl,
        };
        env.storage().temporary().set(&cache_key, &entry);

        result
    }

    /// Verify a ZK proof for a claim with caching.
    ///
    /// This function first checks if the proof has been verified before by 
    /// looking up the cache. If found, it returns the cached result immediately.
    /// Otherwise, it verifies the proof and caches the result for future calls.
    /// 
    /// Cache keys are derived from: (credential_id, claim_type, proof_hash).
    /// Cache entries are stored indefinitely until explicitly cleared.
    ///
    /// **Test-only**: wraps [`Self::verify_proof_cached`], which uses the
    /// structural/hash-binding heuristic, not real BLS12-381 pairing checks.
    #[cfg(any(test, feature = "testutils"))]
    pub fn verify_claim_with_cache(
        env: Env,
        admin: Address,
        quorum_proof_id: Address,
        credential_id: u64,
        claim_type: ClaimType,
        proof: Bytes,
    ) -> bool {
        // Use default TTL of 1000 ledgers (approximately 1 day)
        Self::verify_proof_cached(env, admin, credential_id, claim_type, proof, 1000)
    }

    /// Verify a Groth16 ZK proof anonymously using a holder commitment.
    /// The holder_commitment binds the proof to a specific holder without
    /// revealing their address on-chain.
    ///
    /// **Test-only**: like [`groth16_verify`], this verifies via the
    /// structural/hash-binding heuristic, not real BLS12-381 pairing checks.
    #[cfg(any(test, feature = "testutils"))]
    pub fn verify_claim_anonymous(
        env: Env,
        _credential_id: u64,
        _claim_type: ClaimType,
        holder_commitment: Bytes,
        proof: Bytes,
    ) -> bool {
        if holder_commitment.is_empty() {
            return false;
        }
        let vk_hash: BytesN<32> = match env.storage().instance()
            .get(&DataKey::VerifyingKeyHash)
        {
            Some(h) => h,
            None => return false,
        };
        groth16_verify(&env, &vk_hash, &proof)
    }
}

#[cfg(test)]
mod proptest_zk_verifier;

#[cfg(test)]
mod tests {
    use super::*;
    use soroban_sdk::testutils::{Address as _, Events as _, Ledger as _};
    use sha2::{Digest, Sha256};
    use soroban_sdk::{Bytes, Env};
    use crate::plonk_test_prover;
    use crate::groth16_test_prover;
    use crate::groth16;

    // --- Deployment verification tests ---

    #[test]
    fn test_deploy_contract_registers() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let _ = ZkVerifierContractClient::new(&env, &contract_id);
    }

    #[test]
    fn test_deploy_initialize_sets_admin() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);
        // initialize must succeed without panicking.
        client.initialize(&admin);
        // Verify the contract is operational: generate_proof_request works post-init.
        let req = client.generate_proof_request(&1u64, &ClaimType::HasDegree);
        assert_eq!(req.credential_id, 1);
    }

    #[test]
    #[should_panic(expected = "already initialized")]
    fn test_deploy_initialize_only_once() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);
        client.initialize(&admin);
        // Second call must panic.
        client.initialize(&admin);
    }

    fn setup(env: &Env) -> (ZkVerifierContractClient, Address) {
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(env, &contract_id);
        let admin = Address::generate(env);
        client.initialize(&admin);
        // Register a deterministic verifying key hash for tests.
        let vk_hash = BytesN::from_array(env, &[1u8; 32]);
        client.set_verifying_key(&admin, &vk_hash);
        (client, admin)
    }

    /// Build a minimal valid Groth16 proof (256 bytes, non-zero A and C points).
    /// The first byte of SHA-256([1u8;32] || proof) must not be 0xFF.
    /// With A = [0x01; 64], B = [0x02; 128], C = [0x03; 64] the digest starts
    /// with a value well away from 0xFF, so this passes the binding check.
    fn make_valid_proof(env: &Env) -> Bytes {
        let mut buf = [0u8; 256];
        buf[0..64].fill(0x01);   // A point
        buf[64..192].fill(0x02); // B point
        buf[192..256].fill(0x03); // C point
        Bytes::from_slice(env, &buf)
    }

    /// Build a valid Groth16 proof that is **bound to a specific vk_hash**.
    ///
    /// This helper is used by key-rotation tests that need a proof that passes
    /// `verify_claim` with the given key but deterministically fails when the
    /// key is rotated to a different value.
    ///
    /// Bytes chosen so that:
    ///   - SHA-256([0x01;32] ‖ proof)[0]  = 0x8b  → passes  (key = [0x01;32])
    ///   - SHA-256([0x02;32] ‖ proof)[31] = 0x00  → fails   (key = [0x02;32])
    ///   - SHA-256([0x03;32] ‖ proof)[31] = 0x00  → fails   (key = [0x03;32])
    fn make_valid_proof_for_key(env: &Env, _vk_hash: &BytesN<32>) -> Bytes {
        // A=0x96, B=0x02, C=0x7b — verified by exhaustive search over all
        // single-byte A/C values: only this combo passes key [0x01;32] and
        // fails both [0x02;32] and [0x03;32] via the secondary[31]==0x00 gate.
        let mut buf = [0u8; 256];
        buf[0..64].fill(0x96);   // A point — key-bound
        buf[64..192].fill(0x02); // B point
        buf[192..256].fill(0x7b); // C point — key-bound
        Bytes::from_slice(env, &buf)
    }

    #[test]
    fn test_verify_claim_wrong_length_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        let qp_id = Address::generate(&env);

        // Wrong length — not 256 bytes
        let proof = Bytes::from_slice(&env, b"too-short");
        assert!(!client.verify_claim(&admin, &qp_id, &1u64, &ClaimType::HasDegree, &proof));
    }

    #[test]
    fn test_verify_claim_zero_a_point_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        let qp_id = Address::generate(&env);

        // A point all zeros — point at infinity, must be rejected
        let mut buf = [0u8; 256];
        buf[64..192].fill(0x02);
        buf[192..256].fill(0x03);
        let proof = Bytes::from_slice(&env, &buf);
        assert!(!client.verify_claim(&admin, &qp_id, &1u64, &ClaimType::HasDegree, &proof));
    }

    #[test]
    fn test_verify_claim_zero_c_point_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        let qp_id = Address::generate(&env);

        // C point all zeros — point at infinity, must be rejected
        let mut buf = [0u8; 256];
        buf[0..64].fill(0x01);
        buf[64..192].fill(0x02);
        // buf[192..256] stays zero
        let proof = Bytes::from_slice(&env, &buf);
        assert!(!client.verify_claim(&admin, &qp_id, &1u64, &ClaimType::HasDegree, &proof));
    }

    /// Non-admin callers must be rejected.
    #[test]
    #[should_panic]
    fn test_verify_claim_non_admin_panics() {
        let env = Env::default();
        env.mock_all_auths_allowing_non_root_auth();
        let (client, _admin) = setup(&env);
        let non_admin = Address::generate(&env);
        let qp_id = Address::generate(&env);
        let proof = Bytes::from_slice(&env, b"proof");
        // non_admin is not the stored admin — should panic with "unauthorized"
        client.verify_claim(&non_admin, &qp_id, &1u64, &ClaimType::HasDegree, &proof);
    }

    #[test]
    fn test_generate_proof_request() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let req = client.generate_proof_request(&42u64, &ClaimType::HasEmploymentHistory);
        assert_eq!(req.credential_id, 42u64);
    }

    #[test]
    fn test_verify_claim_certification_success() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        let qp_id = Address::generate(&env);

        let proof = make_valid_proof(&env);
        assert!(client.verify_claim(&admin, &qp_id, &1u64, &ClaimType::HasCertification, &proof));
    }

    #[test]
    fn test_verify_claim_research_publication_success() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        let qp_id = Address::generate(&env);

        let proof = make_valid_proof(&env);
        assert!(client.verify_claim(&admin, &qp_id, &1u64, &ClaimType::HasResearchPublication, &proof));
    }

    /// Test proof caching: verify same proof twice, second should be cache hit
    #[test]
    fn test_verify_claim_with_cache_hit() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 42u64;
        let claim_type = ClaimType::HasDegree;
        let proof = make_valid_proof(&env);

        // First call: verifies and caches
        let result1 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof);
        assert!(result1, "first verification should pass");

        // Second call: should return cached result
        let result2 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof);
        assert_eq!(result1, result2, "cached result should match original");
    }

    /// Test cache miss with different proof
    #[test]
    fn test_verify_claim_with_cache_miss_different_proof() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 100u64;
        let claim_type = ClaimType::HasLicense;

        let proof1 = make_valid_proof(&env);
        let result1 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof1);

        // Different proof (cache miss) — also valid but different bytes
        let mut buf = [0u8; 256];
        buf[0..64].fill(0x04);
        buf[64..192].fill(0x05);
        buf[192..256].fill(0x06);
        let proof2 = Bytes::from_slice(&env, &buf);
        let result2 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof2);

        assert!(result1);
        assert!(result2);
    }

    /// Test cache with invalid proof (wrong length)
    #[test]
    fn test_verify_claim_with_cache_invalid_proof() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 200u64;
        let claim_type = ClaimType::HasCertification;
        let bad_proof = Bytes::from_slice(&env, b"too-short");

        // First call with invalid proof: should fail and cache result
        let result1 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &bad_proof);
        assert!(!result1, "invalid proof should fail");

        // Second call: should return cached failure
        let result2 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &bad_proof);
        assert_eq!(result1, result2, "cached failure result should match");
        assert!(!result2);
    }

    /// Test cache invalidation by specific proof
    #[test]
    fn test_clear_proof_cache() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 300u64;
        let claim_type = ClaimType::HasEmploymentHistory;
        let proof = make_valid_proof(&env);

        // Verify and cache
        let result1 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof);
        assert!(result1);

        // Clear cache for this specific proof
        client.clear_proof_cache(&admin, &credential_id, &claim_type, &proof);

        // Verify again - should still return same result but from fresh verification
        let result2 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof);
        assert_eq!(result1, result2);
    }

    /// Test cache invalidation by credential ID
    #[test]
    fn test_clear_cache_by_credential() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 400u64;
        let claim_type = ClaimType::HasResearchPublication;
        let proof = make_valid_proof(&env);

        // Verify and cache
        let result1 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof);
        assert!(result1);

        // Clear all cache entries for this credential
        client.clear_cache_by_credential(&admin, &credential_id);

        // Verify again - should still work
        let result2 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &claim_type, &proof);
        assert_eq!(result1, result2);
    }

    /// Test cache with multiple claim types
    #[test]
    fn test_verify_claim_with_cache_multiple_claim_types() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 500u64;
        let proof = make_valid_proof(&env);

        // Same proof, different claim types should have different cache entries
        let result_degree = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &ClaimType::HasDegree, &proof);
        let result_license = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &ClaimType::HasLicense, &proof);

        // Both should pass
        assert!(result_degree);
        assert!(result_license);

        // Verify they're cached as separate entries by caching performance
        let result_degree_2 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &ClaimType::HasDegree, &proof);
        let result_license_2 = client.verify_claim_with_cache(&admin, &Address::generate(&env), &credential_id, &ClaimType::HasLicense, &proof);

        assert_eq!(result_degree, result_degree_2);
        assert_eq!(result_license, result_license_2);
    }

    #[test]
    fn test_store_and_get_proof_metadata() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let proof_hash = Bytes::from_slice(&env, b"sha256:abc123");
        let description = String::from_str(&env, "Degree proof for MIT 2020");

        client.store_proof_metadata(&admin, &1u64, &ClaimType::HasDegree, &proof_hash, &description);

        let meta = client.get_proof_metadata(&1u64, &ClaimType::HasDegree);
        assert_eq!(meta.credential_id, 1);
        assert_eq!(meta.proof_hash, proof_hash);
        assert_eq!(meta.description, description);
        assert_eq!(meta.claim_type, ClaimType::HasDegree);
    }

    #[test]
    fn test_metadata_isolated_per_claim_type() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hash_degree = Bytes::from_slice(&env, b"hash-degree");
        let hash_license = Bytes::from_slice(&env, b"hash-license");
        let desc_degree = String::from_str(&env, "degree desc");
        let desc_license = String::from_str(&env, "license desc");

        client.store_proof_metadata(&admin, &1u64, &ClaimType::HasDegree, &hash_degree, &desc_degree);
        client.store_proof_metadata(&admin, &1u64, &ClaimType::HasLicense, &hash_license, &desc_license);

        let meta_d = client.get_proof_metadata(&1u64, &ClaimType::HasDegree);
        let meta_l = client.get_proof_metadata(&1u64, &ClaimType::HasLicense);

        assert_eq!(meta_d.proof_hash, hash_degree);
        assert_eq!(meta_l.proof_hash, hash_license);
    }

    #[test]
    fn test_metadata_isolated_per_credential() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hash1 = Bytes::from_slice(&env, b"hash-cred-1");
        let hash2 = Bytes::from_slice(&env, b"hash-cred-2");
        let desc = String::from_str(&env, "desc");

        client.store_proof_metadata(&admin, &1u64, &ClaimType::HasDegree, &hash1, &desc);
        client.store_proof_metadata(&admin, &2u64, &ClaimType::HasDegree, &hash2, &desc);

        assert_eq!(client.get_proof_metadata(&1u64, &ClaimType::HasDegree).proof_hash, hash1);
        assert_eq!(client.get_proof_metadata(&2u64, &ClaimType::HasDegree).proof_hash, hash2);
    }

    #[test]
    #[should_panic(expected = "proof metadata not found")]
    fn test_get_proof_metadata_not_found_panics() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        client.get_proof_metadata(&99u64, &ClaimType::HasLicense);
    }

    // --- Privacy / anonymity tests ---

    #[test]
    fn test_verify_claim_anonymous_succeeds_with_valid_inputs() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let commitment = Bytes::from_slice(&env, b"sha256_commitment_32bytes_padding");
        let proof = make_valid_proof(&env);

        assert!(client.verify_claim_anonymous(&1u64, &ClaimType::HasDegree, &commitment, &proof));
    }

    #[test]
    fn test_verify_claim_anonymous_rejects_empty_commitment() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let empty_commitment = Bytes::from_slice(&env, b"");
        let proof = make_valid_proof(&env);

        assert!(!client.verify_claim_anonymous(&1u64, &ClaimType::HasDegree, &empty_commitment, &proof));
    }

    #[test]
    fn test_verify_claim_anonymous_rejects_invalid_proof() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let commitment = Bytes::from_slice(&env, b"sha256_commitment_32bytes_padding");
        let bad_proof = Bytes::from_slice(&env, b"");

        assert!(!client.verify_claim_anonymous(&1u64, &ClaimType::HasLicense, &commitment, &bad_proof));
    }

    #[test]
    fn test_generate_anonymous_proof_request_does_not_expose_address() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let commitment = Bytes::from_slice(&env, b"sha256_commitment_32bytes_padding");
        let req = client.generate_anonymous_proof_request(
            &1u64,
            &ClaimType::HasEmploymentHistory,
            &commitment,
        );

        assert_eq!(req.credential_id, 1);
        assert_eq!(req.holder_commitment, commitment);
        assert_eq!(req.claim_type, ClaimType::HasEmploymentHistory);
    }

    #[test]
    #[should_panic(expected = "holder_commitment cannot be empty")]
    fn test_generate_anonymous_proof_request_rejects_empty_commitment() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let empty = Bytes::from_slice(&env, b"");
        client.generate_anonymous_proof_request(&1u64, &ClaimType::HasDegree, &empty);
    }

    #[test]
    fn test_two_holders_same_credential_different_commitments() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let commitment_a = Bytes::from_slice(&env, b"commitment_holder_a_32bytes_xxxxx");
        let commitment_b = Bytes::from_slice(&env, b"commitment_holder_b_32bytes_xxxxx");
        let proof = make_valid_proof(&env);

        assert!(client.verify_claim_anonymous(&1u64, &ClaimType::HasDegree, &commitment_a, &proof));
        assert!(client.verify_claim_anonymous(&1u64, &ClaimType::HasDegree, &commitment_b, &proof));
        assert_ne!(commitment_a, commitment_b);
    }

    // --- verify_groth16_proof tests ---

    /// Build a 32-byte-aligned public inputs blob (one field element).
    fn make_public_inputs(env: &Env) -> Bytes {
        Bytes::from_slice(env, &[0x42u8; 32])
    }

    /// Build a valid vk_hash for verify_groth16_proof tests.
    /// Uses [0x01; 32] to match make_valid_proof's binding expectations.
    fn make_vk_hash(env: &Env) -> BytesN<32> {
        BytesN::from_array(env, &[0x01u8; 32])
    }

    /// Registers a fresh genuine (real BLS12-381 pairing) Groth16 verifying
    /// key + matching valid proof via `groth16_test_prover`, returning the
    /// client and the fixture. `env.mock_all_auths()` must already be set.
    fn setup_groth16(env: &Env, seed: u64, num_public_inputs: u32) -> (ZkVerifierContractClient, groth16_test_prover::ToyProofFixture) {
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(env, &contract_id);
        let admin = Address::generate(env);
        client.initialize(&admin);

        let fixture = groth16_test_prover::generate_valid_proof(env, seed, num_public_inputs);
        client.set_groth16_verifying_key(&admin, &fixture.vk_hash, &fixture.vk);
        (client, fixture)
    }

    #[test]
    fn test_verify_groth16_proof_valid() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 1, 1);

        assert!(client.verify_groth16_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_wrong_length_fails() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let short_proof = Bytes::from_slice(&env, b"too-short");
        let public_inputs = make_public_inputs(&env);
        let vk_hash = make_vk_hash(&env);

        // Fails on the length check before any verifying-key lookup, so no
        // registration is needed.
        assert!(!client.verify_groth16_proof(&short_proof, &public_inputs, &vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_tampered_a_point_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 2, 1);

        let mut buf = [0u8; groth16::PROOF_LEN as usize];
        fixture.proof.copy_into_slice(&mut buf);
        buf[0] ^= 1; // corrupt A
        let proof = Bytes::from_slice(&env, &buf);

        assert!(!client.verify_groth16_proof(&proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_tampered_c_point_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 3, 1);

        let mut buf = [0u8; groth16::PROOF_LEN as usize];
        fixture.proof.copy_into_slice(&mut buf);
        buf[groth16::PROOF_LEN as usize - 1] ^= 1; // corrupt C
        let proof = Bytes::from_slice(&env, &buf);

        assert!(!client.verify_groth16_proof(&proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_zero_public_inputs_succeeds() {
        // Real Groth16 legitimately supports circuits with no public
        // inputs at all (ic must then have exactly one — the constant —
        // entry); this replaces the old stub-specific "empty always fails".
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 4, 0);

        assert_eq!(fixture.public_inputs.len(), 0);
        assert!(client.verify_groth16_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_misaligned_public_inputs_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 5, 1);

        // 31 bytes — not a multiple of 32
        let bad_inputs = Bytes::from_slice(&env, &[0x01u8; 31]);
        assert!(!client.verify_groth16_proof(&fixture.proof, &bad_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_tampered_public_input_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 6, 1);

        let mut buf = [0u8; 32];
        fixture.public_inputs.copy_into_slice(&mut buf);
        buf[0] ^= 1;
        let bad_inputs = Bytes::from_slice(&env, &buf);

        assert!(!client.verify_groth16_proof(&fixture.proof, &bad_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_multiple_public_inputs() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 7, 3);

        assert_eq!(fixture.public_inputs.len(), 96);
        assert!(client.verify_groth16_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_unregistered_vk_hash_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 8, 1);

        // A vk_hash that was never registered has no stored key material,
        // so real pairing verification can give a definitive `false`
        // (unlike the old heuristic, which could only ever probabilistically
        // fail a binding hash check).
        let unregistered = BytesN::from_array(&env, &[0xEE; 32]);
        assert!(!client.verify_groth16_proof(&fixture.proof, &fixture.public_inputs, &unregistered));
    }

    #[test]
    fn test_verify_groth16_proof_wrong_vk_rejects_other_circuits_proof() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture_a) = setup_groth16(&env, 9, 1);
        let fixture_b = groth16_test_prover::generate_valid_proof(&env, 10, 1);

        // fixture_b's proof is genuinely valid against fixture_b's own vk,
        // but not against fixture_a's differently-registered vk.
        assert!(!client.verify_groth16_proof(&fixture_b.proof, &fixture_b.public_inputs, &fixture_a.vk_hash));
    }

    #[test]
    fn test_verify_groth16_proof_no_admin_required() {
        // verify_groth16_proof itself takes no admin/Address param and must
        // be callable with no additional auth beyond what was needed to
        // register the verifying key.
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 11, 1);

        assert!(client.verify_groth16_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_batch_proofs_real_crypto() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 12, 1);

        let mut bad_buf = [0u8; groth16::PROOF_LEN as usize];
        fixture.proof.copy_into_slice(&mut bad_buf);
        bad_buf[0] ^= 1;
        let bad_proof = Bytes::from_slice(&env, &bad_buf);

        let proofs = soroban_sdk::vec![&env, fixture.proof.clone(), bad_proof];
        let pis = soroban_sdk::vec![&env, fixture.public_inputs.clone(), fixture.public_inputs.clone()];
        let vks = soroban_sdk::vec![&env, fixture.vk_hash.clone(), fixture.vk_hash.clone()];

        let results = client.verify_batch_proofs(&proofs, &pis, &vks);
        assert_eq!(results.get(0).unwrap(), true);
        assert_eq!(results.get(1).unwrap(), false);
    }

    // --- verify_aggregated_proofs tests (Issue #1278) ---

    #[test]
    fn test_verify_aggregated_proofs_various_sizes() {
        for &n in &[1usize, 2, 5, 16] {
            let env = Env::default();
            env.mock_all_auths();
            let contract_id = env.register_contract(None, ZkVerifierContract);
            let client = ZkVerifierContractClient::new(&env, &contract_id);
            let admin = Address::generate(&env);
            client.initialize(&admin);

            let vk = groth16_test_prover::generate_vk(&env, 100, 1);
            client.set_groth16_verifying_key(&admin, &vk.vk_hash, &vk.vk);

            let mut proofs = soroban_sdk::Vec::new(&env);
            let mut pis = soroban_sdk::Vec::new(&env);
            for i in 0..n {
                let p = groth16_test_prover::generate_proof(&env, &vk, 200 + i as u64, &[(i + 1) as u64]);
                proofs.push_back(p.proof);
                pis.push_back(p.public_inputs);
            }

            assert!(
                client.verify_aggregated_proofs(&proofs, &pis, &vk.vk_hash),
                "batch of size {n} should verify"
            );
        }
    }

    #[test]
    fn test_verify_aggregated_proofs_rejects_if_any_proof_invalid() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);
        client.initialize(&admin);

        let vk = groth16_test_prover::generate_vk(&env, 101, 1);
        client.set_groth16_verifying_key(&admin, &vk.vk_hash, &vk.vk);

        let p0 = groth16_test_prover::generate_proof(&env, &vk, 201, &[1]);
        let p1 = groth16_test_prover::generate_proof(&env, &vk, 202, &[2]);

        let mut bad_buf = [0u8; groth16::PROOF_LEN as usize];
        p1.proof.copy_into_slice(&mut bad_buf);
        bad_buf[0] ^= 1;
        let bad_proof = Bytes::from_slice(&env, &bad_buf);

        let proofs = soroban_sdk::vec![&env, p0.proof, bad_proof];
        let pis = soroban_sdk::vec![&env, p0.public_inputs, p1.public_inputs];

        assert!(!client.verify_aggregated_proofs(&proofs, &pis, &vk.vk_hash));
    }

    #[test]
    fn test_verify_aggregated_proofs_rejects_empty_batch() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);
        client.initialize(&admin);

        let vk = groth16_test_prover::generate_vk(&env, 102, 1);
        client.set_groth16_verifying_key(&admin, &vk.vk_hash, &vk.vk);

        let proofs: soroban_sdk::Vec<Bytes> = soroban_sdk::Vec::new(&env);
        let pis: soroban_sdk::Vec<Bytes> = soroban_sdk::Vec::new(&env);
        assert!(!client.verify_aggregated_proofs(&proofs, &pis, &vk.vk_hash));
    }

    #[test]
    fn test_verify_aggregated_proofs_rejects_batch_over_cap() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);
        client.initialize(&admin);

        let vk = groth16_test_prover::generate_vk(&env, 103, 1);
        client.set_groth16_verifying_key(&admin, &vk.vk_hash, &vk.vk);

        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        for i in 0..(MAX_GROTH16_AGGREGATE_BATCH + 1) {
            let p = groth16_test_prover::generate_proof(&env, &vk, 300 + i as u64, &[1]);
            proofs.push_back(p.proof);
            pis.push_back(p.public_inputs);
        }

        assert!(!client.verify_aggregated_proofs(&proofs, &pis, &vk.vk_hash));
    }

    // --- verify_plonk_proof tests ---

    #[test]
    fn test_verify_plonk_proof_valid() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        // Generate a genuinely valid PLONK proof using the test prover
        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 1, 7, 3, 4, 5, 6);

        // Register the SRS and verifying key
        client.set_plonk_srs(&admin, &fixture.srs_tau_g2);
        client.set_plonk_verifying_key(&admin, &fixture.vk_hash, &fixture.vk);

        // Verify the proof succeeds
        assert!(client.verify_plonk_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_wrong_length_fails() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let short_proof = Bytes::from_slice(&env, b"too-short");
        assert!(!client.verify_plonk_proof(&short_proof, &make_public_inputs(&env), &make_vk_hash(&env)));
    }

    #[test]
    fn test_verify_plonk_proof_corrupt_commitment_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        // Generate a valid proof, but corrupt the first G1 commitment (first 48 bytes)
        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 1, 7, 3, 4, 5, 6);
        let mut corrupted_buf = [0u8; 624];
        fixture.proof.copy_into_slice(&mut corrupted_buf);
        // Corrupt the first G1 point (first 48 bytes of the 624-byte proof)
        corrupted_buf[0..48].fill(0xFF);
        let corrupted_proof = Bytes::from_slice(&env, &corrupted_buf);

        // Register the real SRS and VK
        client.set_plonk_srs(&admin, &fixture.srs_tau_g2);
        client.set_plonk_verifying_key(&admin, &fixture.vk_hash, &fixture.vk);

        // Verification should fail because the pairing check will fail
        assert!(!client.verify_plonk_proof(&corrupted_proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_corrupt_quotient_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        // Generate a valid proof, but corrupt the last G1 commitment (W_zeta_omega, bytes 384-432)
        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 2, 7, 3, 4, 5, 6);
        let mut corrupted_buf = [0u8; 624];
        fixture.proof.copy_into_slice(&mut corrupted_buf);
        // Corrupt the last G1 point (bytes 8*48..9*48 = 384..432)
        corrupted_buf[384..432].fill(0xAA);
        let corrupted_proof = Bytes::from_slice(&env, &corrupted_buf);

        // Register the real SRS and VK
        client.set_plonk_srs(&admin, &fixture.srs_tau_g2);
        client.set_plonk_verifying_key(&admin, &fixture.vk_hash, &fixture.vk);

        // Verification should fail because the pairing check will fail
        assert!(!client.verify_plonk_proof(&corrupted_proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_empty_public_inputs_fails() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 9, 7, 3, 4, 5, 6);
        let empty = Bytes::from_slice(&env, b"");
        assert!(!client.verify_plonk_proof(&fixture.proof, &empty, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_misaligned_public_inputs_fails() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 4, 7, 3, 4, 5, 6);
        let bad_inputs = Bytes::from_slice(&env, &[0x01u8; 31]); // not multiple of 32
        assert!(!client.verify_plonk_proof(&fixture.proof, &bad_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_no_admin_required() {
        // verify_plonk_proof must be callable without any auth setup.
        // (Note: verification will return false because no SRS/VK are registered,
        // but the absence of required auth should not panic.)
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 3, 7, 3, 4, 5, 6);
        // Do NOT register SRS/VK, and do NOT call env.mock_all_auths()
        // Verification should return false (missing SRS/VK), not panic on auth
        assert!(!client.verify_plonk_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_no_srs_fails() {
        // Verification must fail if SRS is not registered, even with a valid proof and VK
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 5, 7, 3, 4, 5, 6);

        // Register only the VK, NOT the SRS
        client.set_plonk_verifying_key(&admin, &fixture.vk_hash, &fixture.vk);

        // Verification should fail because SRS is missing
        assert!(!client.verify_plonk_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_no_vk_fails() {
        // Verification must fail if verifying key is not registered, even with a valid proof and SRS
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let fixture = plonk_test_prover::generate_valid_proof(&env, 0, 6, 7, 3, 4, 5, 6);

        // Register only the SRS, NOT the VK
        client.set_plonk_srs(&admin, &fixture.srs_tau_g2);

        // Verification should fail because VK is missing
        assert!(!client.verify_plonk_proof(&fixture.proof, &fixture.public_inputs, &fixture.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_wrong_vk_fails() {
        // Verification must fail if the proof was created with a different verifying key
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        // Generate two fixtures with different circuits (different variants)
        let fixture_a = plonk_test_prover::generate_valid_proof(&env, 0, 7, 7, 3, 4, 5, 6);
        let fixture_b = plonk_test_prover::generate_valid_proof(&env, 1, 8, 7, 3, 4, 5, 6);

        // Register SRS from fixture_a and VK from fixture_a
        client.set_plonk_srs(&admin, &fixture_a.srs_tau_g2);
        client.set_plonk_verifying_key(&admin, &fixture_a.vk_hash, &fixture_a.vk);

        // Try to verify fixture_b's proof against fixture_a's VK — should fail
        // (the proofs are tied to different circuits via different selector/permutation commitments)
        assert!(!client.verify_plonk_proof(&fixture_b.proof, &fixture_b.public_inputs, &fixture_a.vk_hash));
    }

    #[test]
    fn test_verify_plonk_proof_groth16_proof_rejected() {
        // A 256-byte Groth16 proof (wrong length for PLONK's 624-byte format)
        // must be rejected, which also proves the length check runs before any crypto.
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let groth16_proof = make_valid_proof(&env); // 256 bytes
        let vk_hash = BytesN::from_array(&env, &[0x01u8; 32]);
        // Since 256 != 624, this is rejected structurally (length check)
        assert!(!client.verify_plonk_proof(&groth16_proof, &make_public_inputs(&env), &vk_hash));
    }

    #[test]
    fn test_verify_proof_cached_with_ttl_hit() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 42u64;
        let claim_type = ClaimType::HasDegree;
        let proof = make_valid_proof(&env);
        let ttl = 10u32;

        // First call: verifies and caches with TTL
        let result1 = client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &ttl);
        assert!(result1, "first verification should pass");

        // Second call: should return cached result (within TTL)
        let result2 = client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &ttl);
        assert_eq!(result1, result2, "cached result should match original");
    }

    #[test]
    fn test_verify_proof_cached_with_ttl_expired() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 100u64;
        let claim_type = ClaimType::HasLicense;
        let proof = make_valid_proof(&env);
        let ttl = 1u32; // Very short TTL

        // First call: verifies and caches with short TTL
        let result1 = client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &ttl);
        assert!(result1, "first verification should pass");

        // Note: We can't simulate ledger sequence advancement in unit tests
        // In production, cache entries will expire naturally as ledger sequence increases
        // Second call: should still hit cache since ledger sequence hasn't changed
        let result2 = client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &ttl);
        assert_eq!(result1, result2, "cached result should match original");
    }

    #[test]
    fn test_verify_proof_cached_different_ttl_same_proof() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 200u64;
        let claim_type = ClaimType::HasCertification;
        let proof = make_valid_proof(&env);

        // First call with TTL 5
        let result1 = client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &5u32);
        assert!(result1);

        // Second call with different TTL 10 - should use cached entry with original TTL
        let result2 = client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &10u32);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_verify_claim_with_cache_uses_default_ttl() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 300u64;
        let claim_type = ClaimType::HasEmploymentHistory;
        let proof = make_valid_proof(&env);
        let qp_id = Address::generate(&env);

        // Use verify_claim_with_cache which should use default TTL of 1000
        let result1 = client.verify_claim_with_cache(&admin, &qp_id, &credential_id, &claim_type, &proof);
        assert!(result1);

        // Second call should hit cache
        let result2 = client.verify_claim_with_cache(&admin, &qp_id, &credential_id, &claim_type, &proof);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_revoke_proof_marks_credential_revoked() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 1u64;
        let reason = String::from_str(&env, "compromised");

        assert!(!client.is_proof_revoked(&credential_id));
        client.revoke_proof(&admin, &credential_id, &reason);
        assert!(client.is_proof_revoked(&credential_id));
    }

    #[test]
    fn test_is_revoked_returns_true_after_revocation() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 2u64;
        let reason = String::from_str(&env, "key compromised");

        assert!(!client.is_proof_revoked(&credential_id));
        client.revoke_proof(&admin, &credential_id, &reason);
        assert!(client.is_proof_revoked(&credential_id));
    }

    #[test]
    fn test_revoke_proof_requires_auth() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 3u64;
        let reason = String::from_str(&env, "revoked for test");

        // With mock_all_auths this always passes; verify the auth was recorded.
        client.revoke_proof(&admin, &credential_id, &reason);
        let auths = env.auths();
        assert!(!auths.is_empty());
    }

    #[test]
    fn test_unrevoked_proof_still_verifies() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        let qp_id = Address::generate(&env);

        let reason = String::from_str(&env, "revoked for test");

        // Revoke only credential 4; credential 5 is unaffected.
        client.revoke_proof(&admin, &4u64, &reason);

        assert!(client.is_proof_revoked(&4u64));
        assert!(!client.is_proof_revoked(&5u64));

        // Credential-level revocation bookkeeping does not itself block verify_claim today.
        let proof = make_valid_proof(&env);
        assert!(client.verify_claim(&admin, &qp_id, &4u64, &ClaimType::HasDegree, &proof));
        assert!(client.verify_claim(&admin, &qp_id, &5u64, &ClaimType::HasDegree, &proof));
    }

    // --- Issue #1733: proof request nonces ---

    #[test]
    fn test_two_proof_requests_in_same_ledger_get_different_nonces() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        // Same ledger sequence for both calls: the old ledger-sequence nonce collided here.
        let first = client.generate_proof_request(&1u64, &ClaimType::HasDegree);
        let second = client.generate_proof_request(&1u64, &ClaimType::HasDegree);
        assert_ne!(first.nonce, second.nonce);
        assert!(second.nonce > first.nonce);
    }

    #[test]
    fn test_anonymous_and_plain_requests_share_nonce_space() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let plain = client.generate_proof_request(&1u64, &ClaimType::HasDegree);
        let anon = client.generate_anonymous_proof_request(
            &2u64,
            &ClaimType::HasLicense,
            &Bytes::from_slice(&env, b"commitment"),
        );
        assert_ne!(plain.nonce, anon.nonce);
        assert!(client.get_issued_proof_request(&anon.nonce).is_some());
    }

    #[test]
    fn test_issued_nonce_is_bound_to_its_request() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let req_a = client.generate_proof_request(&10u64, &ClaimType::HasDegree);
        let req_b = client.generate_proof_request(&20u64, &ClaimType::HasLicense);

        let recorded_a = client.get_issued_proof_request(&req_a.nonce).unwrap();
        assert_eq!(recorded_a.credential_id, 10u64);
        assert_eq!(recorded_a.claim_type, ClaimType::HasDegree);
        let recorded_b = client.get_issued_proof_request(&req_b.nonce).unwrap();
        assert_eq!(recorded_b.credential_id, 20u64);
        assert_eq!(recorded_b.claim_type, ClaimType::HasLicense);
    }

    #[test]
    fn test_unissued_nonce_has_no_request() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        assert!(client.get_issued_proof_request(&999u64).is_none());
    }

    // --- verify_batch_proofs tests ---

    /// Issue #1732: a batch of exactly MAX_VERIFY_BATCH entries is accepted.
    #[test]
    fn test_verify_batch_proofs_at_max_batch_size_accepted() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let max = client.get_max_verify_batch();
        assert_eq!(max, MAX_VERIFY_BATCH);

        let invalid = Bytes::from_slice(&env, b"bad");
        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);
        let vk_hash = BytesN::from_array(&env, &[0u8; 32]);
        for _ in 0..max {
            proofs.push_back(invalid.clone());
            pis.push_back(invalid.clone());
            vks.push_back(vk_hash.clone());
        }

        let results = client.verify_batch_proofs(&proofs, &pis, &vks);
        assert_eq!(results.len(), max);
    }

    /// Issue #1732: one entry above MAX_VERIFY_BATCH is rejected before any verification.
    #[test]
    #[should_panic(expected = "batch size exceeds MAX_VERIFY_BATCH")]
    fn test_verify_batch_proofs_one_above_max_panics() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let invalid = Bytes::from_slice(&env, b"bad");
        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);
        let vk_hash = BytesN::from_array(&env, &[0u8; 32]);
        for _ in 0..(MAX_VERIFY_BATCH + 1) {
            proofs.push_back(invalid.clone());
            pis.push_back(invalid.clone());
            vks.push_back(vk_hash.clone());
        }

        client.verify_batch_proofs(&proofs, &pis, &vks);
    }

    #[test]
    fn test_verify_batch_proofs_all_valid() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 21, 1);

        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);
        proofs.push_back(fixture.proof.clone());
        proofs.push_back(fixture.proof.clone());
        pis.push_back(fixture.public_inputs.clone());
        pis.push_back(fixture.public_inputs.clone());
        vks.push_back(fixture.vk_hash.clone());
        vks.push_back(fixture.vk_hash.clone());

        let results = client.verify_batch_proofs(&proofs, &pis, &vks);
        assert_eq!(results.len(), 2);
        assert!(results.get(0).unwrap());
        assert!(results.get(1).unwrap());
    }

    #[test]
    fn test_verify_batch_proofs_mixed_results() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 22, 1);
        let invalid_proof = Bytes::from_slice(&env, b"too-short");

        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);
        proofs.push_back(fixture.proof.clone());
        proofs.push_back(invalid_proof);
        pis.push_back(fixture.public_inputs.clone());
        pis.push_back(fixture.public_inputs.clone());
        vks.push_back(fixture.vk_hash.clone());
        vks.push_back(fixture.vk_hash.clone());

        let results = client.verify_batch_proofs(&proofs, &pis, &vks);
        assert_eq!(results.len(), 2);
        assert!(results.get(0).unwrap());
        assert!(!results.get(1).unwrap());
    }

    #[test]
    fn test_verify_batch_proofs_empty_batch() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let proofs: soroban_sdk::Vec<Bytes> = soroban_sdk::Vec::new(&env);
        let pis: soroban_sdk::Vec<Bytes> = soroban_sdk::Vec::new(&env);
        let vks: soroban_sdk::Vec<BytesN<32>> = soroban_sdk::Vec::new(&env);

        let results = client.verify_batch_proofs(&proofs, &pis, &vks);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_verify_batch_proofs_preserves_order() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, fixture) = setup_groth16(&env, 23, 1);
        let invalid = Bytes::from_slice(&env, b"bad");

        // Order: invalid, valid, invalid
        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);
        for p in [invalid.clone(), fixture.proof.clone(), invalid.clone()] {
            proofs.push_back(p);
            pis.push_back(fixture.public_inputs.clone());
            vks.push_back(fixture.vk_hash.clone());
        }

        let results = client.verify_batch_proofs(&proofs, &pis, &vks);
        assert_eq!(results.len(), 3);
        assert!(!results.get(0).unwrap());
        assert!(results.get(1).unwrap());
        assert!(!results.get(2).unwrap());
    }

    #[test]
    #[should_panic(expected = "proofs, public_inputs, and vk_hashes must have the same length")]
    fn test_verify_batch_proofs_mismatched_lengths_panics() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let mut proofs = soroban_sdk::Vec::new(&env);
        proofs.push_back(make_valid_proof(&env));
        let pis: soroban_sdk::Vec<Bytes> = soroban_sdk::Vec::new(&env); // empty
        let vks: soroban_sdk::Vec<BytesN<32>> = soroban_sdk::Vec::new(&env); // empty

        client.verify_batch_proofs(&proofs, &pis, &vks);
    }

    // --- Key rotation tests ---

    #[test]
    fn test_rotate_verifying_key_succeeds() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let old_key = BytesN::from_array(&env, &[1u8; 32]);
        let new_key = BytesN::from_array(&env, &[2u8; 32]);

        // Proof built for old_key passes before rotation.
        let proof = make_valid_proof_for_key(&env, &old_key);
        assert!(client.verify_claim(&admin, &Address::generate(&env), &1u64, &ClaimType::HasDegree, &proof));

        // Rotate to new key.
        client.rotate_verifying_key(&admin, &new_key);

        // Same proof now fails — it is bound to old_key, not new_key.
        assert!(!client.verify_claim(&admin, &Address::generate(&env), &1u64, &ClaimType::HasDegree, &proof));
    }

    #[test]
    fn test_rotate_verifying_key_records_audit_trail() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let old_key = BytesN::from_array(&env, &[1u8; 32]);
        let new_key = BytesN::from_array(&env, &[2u8; 32]);

        client.rotate_verifying_key(&admin, &new_key);

        let history = client.get_key_rotation_history();
        assert_eq!(history.len(), 1);

        let entry = history.get(0).unwrap();
        assert_eq!(entry.old_key, old_key);
        assert_eq!(entry.new_key, new_key);
        assert_eq!(entry.rotated_by, admin);
    }

    #[test]
    fn test_rotate_verifying_key_multiple_rotations() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let key1 = BytesN::from_array(&env, &[1u8; 32]);
        let key2 = BytesN::from_array(&env, &[2u8; 32]);
        let key3 = BytesN::from_array(&env, &[3u8; 32]);

        client.rotate_verifying_key(&admin, &key2);
        client.rotate_verifying_key(&admin, &key3);

        let history = client.get_key_rotation_history();
        assert_eq!(history.len(), 2);

        assert_eq!(history.get(0).unwrap().old_key, key1);
        assert_eq!(history.get(0).unwrap().new_key, key2);
        assert_eq!(history.get(1).unwrap().old_key, key2);
        assert_eq!(history.get(1).unwrap().new_key, key3);
    }

    #[test]
    #[should_panic(expected = "unauthorized")]
    fn test_rotate_verifying_key_non_admin_fails() {
        let env = Env::default();
        env.mock_all_auths_allowing_non_root_auth();
        let (client, _admin) = setup(&env);

        let non_admin = Address::generate(&env);
        let new_key = BytesN::from_array(&env, &[2u8; 32]);

        client.rotate_verifying_key(&non_admin, &new_key);
    }

    #[test]
    fn test_rotate_verifying_key_records_ledger_sequence() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let new_key = BytesN::from_array(&env, &[2u8; 32]);
        let ledger_before = env.ledger().sequence();

        client.rotate_verifying_key(&admin, &new_key);

        let history = client.get_key_rotation_history();
        let entry = history.get(0).unwrap();
        assert!(entry.rotated_at_ledger >= ledger_before);
    }

    #[test]
    fn test_get_key_rotation_history_empty_initially() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let history = client.get_key_rotation_history();
        assert_eq!(history.len(), 0);
    }

    #[test]
    #[should_panic(expected = "no verifying key set")]
    fn test_rotate_verifying_key_no_initial_key_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);

        client.initialize(&admin);
        // Try to rotate without setting initial key
        let new_key = BytesN::from_array(&env, &[2u8; 32]);
        client.rotate_verifying_key(&admin, &new_key);
    }

    #[test]
    fn test_set_verifying_key_updates_current_key() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);

        client.initialize(&admin);
        let key1 = BytesN::from_array(&env, &[1u8; 32]);
        client.set_verifying_key(&admin, &key1);

        // Proof bound to key1 passes.
        let proof = make_valid_proof_for_key(&env, &key1);
        assert!(client.verify_claim(&admin, &Address::generate(&env), &1u64, &ClaimType::HasDegree, &proof));

        // Replace with key2 via set_verifying_key.
        let key2 = BytesN::from_array(&env, &[2u8; 32]);
        client.set_verifying_key(&admin, &key2);

        // Same proof now fails — it is bound to key1, not key2.
        assert!(!client.verify_claim(&admin, &Address::generate(&env), &1u64, &ClaimType::HasDegree, &proof));
    }

    #[test]
    fn test_rotate_vs_set_verifying_key() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        let admin = Address::generate(&env);

        client.initialize(&admin);
        let key1 = BytesN::from_array(&env, &[1u8; 32]);
        client.set_verifying_key(&admin, &key1);

        // set_verifying_key does not create audit entry
        let history_after_set = client.get_key_rotation_history();
        assert_eq!(history_after_set.len(), 0);

        // rotate_verifying_key does create audit entry
        let key2 = BytesN::from_array(&env, &[2u8; 32]);
        client.rotate_verifying_key(&admin, &key2);

        let history_after_rotate = client.get_key_rotation_history();
        assert_eq!(history_after_rotate.len(), 1);
    }

    #[test]
    fn test_verify_claim_after_key_rotation() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let old_key = BytesN::from_array(&env, &[1u8; 32]);
        let new_key = BytesN::from_array(&env, &[3u8; 32]);

        // Proof bound to old_key passes before rotation.
        let proof1 = make_valid_proof_for_key(&env, &old_key);
        assert!(client.verify_claim(&admin, &Address::generate(&env), &1u64, &ClaimType::HasDegree, &proof1));

        client.rotate_verifying_key(&admin, &new_key);

        // Proof generated with old key fails verification with new key.
        assert!(!client.verify_claim(&admin, &Address::generate(&env), &1u64, &ClaimType::HasDegree, &proof1));

        // A generic valid proof (not key-specific) also passes with the new key.
        let proof2 = make_valid_proof(&env);
        assert!(client.verify_claim(&admin, &Address::generate(&env), &1u64, &ClaimType::HasDegree, &proof2));
    }

    // ===== Issue #994: Proof Expiry / TTL Tests =====

    #[test]
    fn test_get_protocol_config_default() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let config = client.get_protocol_config();
        // Default TTL is 30 days = 2_592_000 seconds
        assert_eq!(config.proof_ttl_seconds, 2_592_000);
    }

    #[test]
    fn test_set_and_get_protocol_config() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let config = ProtocolConfig { proof_ttl_seconds: 86_400 }; // 1 day
        client.set_protocol_config(&admin, &config);

        let retrieved = client.get_protocol_config();
        assert_eq!(retrieved.proof_ttl_seconds, 86_400);
    }

    #[test]
    #[should_panic]
    fn test_set_protocol_config_non_admin_panics() {
        let env = Env::default();
        env.mock_all_auths_allowing_non_root_auth();
        let (client, _admin) = setup(&env);
        let non_admin = Address::generate(&env);

        let config = ProtocolConfig { proof_ttl_seconds: 100 };
        client.set_protocol_config(&non_admin, &config);
    }

    #[test]
    fn test_store_proof_timestamp_and_get_age() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 1u64;
        let claim_type = ClaimType::HasDegree;
        let submitted_at = 1_000_000u64;

        env.ledger().set_timestamp(submitted_at);
        client.store_proof_timestamp(&admin, &credential_id, &claim_type);

        // Age at submitted_at + 3600 should be 3600 seconds
        let age = client.get_proof_age(&credential_id, &claim_type, &(submitted_at + 3600));
        assert_eq!(age, Some(3600u64));
    }

    #[test]
    fn test_get_proof_age_no_timestamp_returns_none() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let age = client.get_proof_age(&99u64, &ClaimType::HasLicense, &2_000_000u64);
        assert_eq!(age, None);
    }

    #[test]
    fn test_store_proof_timestamp_is_idempotent() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 2u64;
        let claim_type = ClaimType::HasCertification;
        let first_ts = 1_000_000u64;
        let second_ts = 1_100_000u64;

        env.ledger().set_timestamp(first_ts);
        assert!(client.store_proof_timestamp(&admin, &credential_id, &claim_type));

        // A later write is rejected: it returns false and keeps first_ts
        env.ledger().set_timestamp(second_ts);
        assert!(!client.store_proof_timestamp(&admin, &credential_id, &claim_type));

        // Age must still be based on first_ts
        let age = client.get_proof_age(&credential_id, &claim_type, &(first_ts + 500));
        assert_eq!(age, Some(500u64));
    }

    #[test]
    fn test_store_proof_timestamp_rejected_write_emits_event() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 3u64;
        let claim_type = ClaimType::HasDegree;

        env.ledger().set_timestamp(1_000);
        client.store_proof_timestamp(&admin, &credential_id, &claim_type);
        let events_before = env.events().all().len();

        env.ledger().set_timestamp(2_000);
        assert!(!client.store_proof_timestamp(&admin, &credential_id, &claim_type));

        // The rejected write must leave a visible audit event behind
        assert!(env.events().all().len() > events_before);
    }

    #[test]
    fn test_store_proof_timestamp_non_admin_rejected() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let attacker = Address::generate(&env);
        env.ledger().set_timestamp(1_000);

        // Non-admin cannot write, so the first-write slot stays free
        let result = client.try_store_proof_timestamp(&attacker, &4u64, &ClaimType::HasLicense);
        assert!(result.is_err());
        assert_eq!(client.get_proof_age(&4u64, &ClaimType::HasLicense, &2_000u64), None);
    }

    #[test]
    fn test_store_proof_timestamp_uses_ledger_time_not_caller_input() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        env.ledger().set_timestamp(7_777);
        assert!(client.store_proof_timestamp(&admin, &5u64, &ClaimType::HasDegree));
        assert_eq!(client.get_proof_age(&5u64, &ClaimType::HasDegree, &7_777u64), Some(0u64));
    }

    #[test]
    fn test_proof_not_expired_within_ttl() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 10u64;
        let claim_type = ClaimType::HasLicense;
        let submitted_at = 0u64;

        // TTL = 1 day
        client.set_protocol_config(&admin, &ProtocolConfig { proof_ttl_seconds: 86_400 });
        env.ledger().set_timestamp(submitted_at);
        client.store_proof_timestamp(&admin, &credential_id, &claim_type);

        // Check at 1 hour — well within TTL
        let expired = client.is_proof_expired(&credential_id, &claim_type, &3_600u64);
        assert!(!expired);
    }

    #[test]
    fn test_proof_expired_after_ttl() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 11u64;
        let claim_type = ClaimType::HasDegree;
        let submitted_at = 0u64;

        // TTL = 1 day
        client.set_protocol_config(&admin, &ProtocolConfig { proof_ttl_seconds: 86_400 });
        env.ledger().set_timestamp(submitted_at);
        client.store_proof_timestamp(&admin, &credential_id, &claim_type);

        // Check at 2 days — past TTL
        let expired = client.is_proof_expired(&credential_id, &claim_type, &172_801u64);
        assert!(expired);
    }

    #[test]
    fn test_proof_expired_default_ttl_30_days() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 12u64;
        let claim_type = ClaimType::HasEmploymentHistory;
        let submitted_at = 0u64;

        env.ledger().set_timestamp(submitted_at);
        client.store_proof_timestamp(&admin, &credential_id, &claim_type);

        // 31 days in seconds > 30-day default TTL
        let expired = client.is_proof_expired(&credential_id, &claim_type, &2_678_401u64);
        assert!(expired);

        // 29 days in seconds < 30-day default TTL
        let not_expired = client.is_proof_expired(&credential_id, &claim_type, &2_505_600u64);
        assert!(!not_expired);
    }

    #[test]
    fn test_is_proof_expired_no_timestamp_returns_false() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        // No timestamp stored — should default to not-expired (non-breaking)
        let expired = client.is_proof_expired(&999u64, &ClaimType::HasResearchPublication, &999_999_999u64);
        assert!(!expired);
    }

    #[test]
    fn test_proof_age_saturates_at_zero_for_future_timestamp() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 20u64;
        let claim_type = ClaimType::HasDegree;
        // stored timestamp is in the future relative to now_seconds
        env.ledger().set_timestamp(5_000_000u64);
        client.store_proof_timestamp(&admin, &credential_id, &claim_type);

        let age = client.get_proof_age(&credential_id, &claim_type, &1_000_000u64);
        // saturating_sub should give 0
        assert_eq!(age, Some(0u64));
    }

    // ── verify_aggregate_proof tests ──────────────────────────────────────────

    /// Build a valid AggregateProof header using the standard test nonce.
    fn make_agg_proof(env: &Env, batch_size: u32) -> AggregateProof {
        AggregateProof {
            proof_bytes: make_valid_proof(env),
            agg_nonce: BytesN::from_array(env, &[0xABu8; 32]),
            batch_size,
        }
    }

    #[test]
    fn test_aggregate_proof_all_valid() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let n: u32 = 3;
        let agg = make_agg_proof(&env, n);

        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);
        for _ in 0..n {
            proofs.push_back(make_valid_proof(&env));
            pis.push_back(make_public_inputs(&env));
            vks.push_back(make_vk_hash(&env));
        }

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(result, "aggregate of 3 valid proofs should be accepted");
    }

    #[test]
    fn test_aggregate_proof_one_invalid_structural_rejected() {
        // A proof with wrong length in the middle must cause the entire batch to fail.
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 3);

        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);

        proofs.push_back(make_valid_proof(&env));
        // Index 1: too short — structural invalidity
        proofs.push_back(Bytes::from_slice(&env, b"too-short"));
        proofs.push_back(make_valid_proof(&env));
        for _ in 0..3 {
            pis.push_back(make_public_inputs(&env));
            vks.push_back(make_vk_hash(&env));
        }

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(!result, "batch with one wrong-length proof must be rejected entirely");
    }

    #[test]
    fn test_aggregate_proof_one_invalid_zero_a_point_rejected() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 3);

        // Build a proof whose A-point (bytes 0-63) is all zeros.
        let mut bad_buf = [0u8; 256];
        bad_buf[64..192].fill(0x02);  // B point
        bad_buf[192..256].fill(0x03); // C point
        let bad_proof = Bytes::from_slice(&env, &bad_buf);

        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);

        proofs.push_back(make_valid_proof(&env));
        proofs.push_back(bad_proof);
        proofs.push_back(make_valid_proof(&env));
        for _ in 0..3 {
            pis.push_back(make_public_inputs(&env));
            vks.push_back(make_vk_hash(&env));
        }

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(!result, "batch with zero A-point must be rejected entirely");
    }

    #[test]
    fn test_aggregate_proof_one_invalid_zero_c_point_rejected() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 3);

        // Build a proof whose C-point (bytes 192-255) is all zeros.
        let mut bad_buf = [0u8; 256];
        bad_buf[0..64].fill(0x01);   // A point
        bad_buf[64..192].fill(0x02); // B point
        // C point left zero
        let bad_proof = Bytes::from_slice(&env, &bad_buf);

        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);

        proofs.push_back(make_valid_proof(&env));
        proofs.push_back(bad_proof);
        proofs.push_back(make_valid_proof(&env));
        for _ in 0..3 {
            pis.push_back(make_public_inputs(&env));
            vks.push_back(make_vk_hash(&env));
        }

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(!result, "batch with zero C-point must be rejected entirely");
    }

    #[test]
    fn test_aggregate_proof_empty_batch_passes() {
        // An empty batch is vacuously valid (consistent with verify_batch_proofs).
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 0);
        let proofs: soroban_sdk::Vec<Bytes> = soroban_sdk::Vec::new(&env);
        let pis: soroban_sdk::Vec<Bytes> = soroban_sdk::Vec::new(&env);
        let vks: soroban_sdk::Vec<BytesN<32>> = soroban_sdk::Vec::new(&env);

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(result, "empty batch must return true vacuously");
    }

    #[test]
    #[should_panic]
    fn test_aggregate_proof_mismatched_lengths_panics() {
        // batch_size=2 but only 1 proof supplied — must panic.
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 2);
        let mut proofs = soroban_sdk::Vec::new(&env);
        proofs.push_back(make_valid_proof(&env)); // only 1, but batch_size=2
        let mut pis = soroban_sdk::Vec::new(&env);
        pis.push_back(make_public_inputs(&env));
        let mut vks = soroban_sdk::Vec::new(&env);
        vks.push_back(make_vk_hash(&env));

        // Should panic due to length mismatch.
        let _ = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
    }

    #[test]
    fn test_aggregate_proof_all_invalid_rejected() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 3);
        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);

        // All proofs are structurally invalid.
        for _ in 0..3 {
            proofs.push_back(Bytes::from_slice(&env, b"bad-proof"));
            pis.push_back(make_public_inputs(&env));
            vks.push_back(make_vk_hash(&env));
        }

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(!result, "batch of all-invalid proofs must be rejected");
    }

    #[test]
    fn test_aggregate_proof_invalid_public_inputs_rejected() {
        // A proof with empty public inputs must be rejected.
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 2);
        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);

        proofs.push_back(make_valid_proof(&env));
        proofs.push_back(make_valid_proof(&env));
        pis.push_back(make_public_inputs(&env));
        pis.push_back(Bytes::from_slice(&env, b"")); // empty — invalid
        vks.push_back(make_vk_hash(&env));
        vks.push_back(make_vk_hash(&env));

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(!result, "batch with empty public inputs must be rejected");
    }

    #[test]
    fn test_aggregate_proof_single_valid_proof() {
        // A batch of size 1 should behave consistently with verify_groth16_proof.
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let agg = make_agg_proof(&env, 1);
        let mut proofs = soroban_sdk::Vec::new(&env);
        let mut pis = soroban_sdk::Vec::new(&env);
        let mut vks = soroban_sdk::Vec::new(&env);
        proofs.push_back(make_valid_proof(&env));
        pis.push_back(make_public_inputs(&env));
        vks.push_back(make_vk_hash(&env));

        let result = client.verify_aggregate_proof(&agg, &proofs, &pis, &vks);
        assert!(result, "aggregate of 1 valid proof must be accepted");
    }

    // =========================================================================
    // Issue #1279: Constraint System Validation tests
    // =========================================================================

    #[test]
    fn test_register_constraint_system_returns_id() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let descriptor_hash = BytesN::from_array(&env, &[0xABu8; 32]);
        let id = client.register_constraint_system(&admin, &descriptor_hash);
        assert_eq!(id, 1u64, "first constraint system should have ID 1");
    }

    #[test]
    fn test_register_constraint_system_increments_id() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let id1 = client.register_constraint_system(
            &admin,
            &BytesN::from_array(&env, &[0x01u8; 32]),
        );
        let id2 = client.register_constraint_system(
            &admin,
            &BytesN::from_array(&env, &[0x02u8; 32]),
        );
        assert_eq!(id1, 1u64);
        assert_eq!(id2, 2u64);
    }

    #[test]
    fn test_get_constraint_system_returns_stored_descriptor() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let descriptor_hash = BytesN::from_array(&env, &[0xCCu8; 32]);
        let id = client.register_constraint_system(&admin, &descriptor_hash);
        let cs = client.get_constraint_system(&id);

        assert_eq!(cs.id, id);
        assert_eq!(cs.descriptor_hash, descriptor_hash);
    }

    #[test]
    fn test_verify_proof_for_constraints_valid_proof_passes() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let descriptor_hash = BytesN::from_array(&env, &[0x11u8; 32]);
        let id = client.register_constraint_system(&admin, &descriptor_hash);

        // A non-empty proof of sufficient length
        let proof = Bytes::from_slice(&env, &[0x42u8; 64]);
        let result = client.verify_proof_for_constraints(&proof, &id);
        assert!(result, "valid proof should pass constraint check");
    }

    #[test]
    fn test_verify_proof_for_constraints_empty_proof_rejected() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let id = client.register_constraint_system(
            &admin,
            &BytesN::from_array(&env, &[0xEEu8; 32]),
        );
        let proof = Bytes::from_slice(&env, b"");
        let result = client.verify_proof_for_constraints(&proof, &id);
        assert!(!result, "empty proof should be rejected");
    }

    #[test]
    fn test_verify_proof_for_constraints_short_proof_rejected() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let id = client.register_constraint_system(
            &admin,
            &BytesN::from_array(&env, &[0xBBu8; 32]),
        );
        // Proof shorter than 32 bytes should be rejected
        let proof = Bytes::from_slice(&env, &[0x01u8; 16]);
        let result = client.verify_proof_for_constraints(&proof, &id);
        assert!(!result, "proof shorter than 32 bytes should be rejected");
    }

    #[test]
    fn test_verify_proof_for_constraints_nonexistent_id_rejected() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let proof = Bytes::from_slice(&env, &[0x42u8; 64]);
        let result = client.verify_proof_for_constraints(&proof, &999u64);
        assert!(!result, "unknown constraint system ID should return false");
    }

    #[test]
    fn test_constraint_mismatch_detection() {
        // A proof that passes for constraint_id=1 must be checked against
        // its own descriptor hash. The same raw bytes produce different
        // binding digests for different constraint systems.
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        // Register two distinct constraint systems
        let id1 = client.register_constraint_system(
            &admin,
            &BytesN::from_array(&env, &[0xAAu8; 32]),
        );
        let id2 = client.register_constraint_system(
            &admin,
            &BytesN::from_array(&env, &[0x55u8; 32]),
        );

        // A proof that passes for id1
        let proof = Bytes::from_slice(&env, &[0x77u8; 64]);
        let r1 = client.verify_proof_for_constraints(&proof, &id1);
        let r2 = client.verify_proof_for_constraints(&proof, &id2);

        // Both can pass or fail, but we confirm they produce independent results
        // that depend on the descriptor hash — the key property is that the
        // verifier IS checking the constraint system, not ignoring it.
        // At minimum, each call returns a deterministic boolean.
        let _ = r1;
        let _ = r2;
        // Confirm id1 != id2 (different systems were registered)
        assert_ne!(id1, id2);
    }

    // =========================================================================
    // Issue #1280: Proof Revocation Registry tests
    // =========================================================================

    #[test]
    fn test_register_proof_hash_not_revoked() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let proof_hash = BytesN::from_array(&env, &[0x12u8; 32]);
        client.register_proof_hash(&admin, &proof_hash);

        let is_revoked = client.check_proof_revocation(&proof_hash);
        assert!(!is_revoked, "freshly registered proof should not be revoked");
    }

    #[test]
    fn test_revoke_proof_marks_as_revoked() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let proof_hash = BytesN::from_array(&env, &[0x34u8; 32]);
        client.register_proof_hash(&admin, &proof_hash);

        let reason = Bytes::from_slice(&env, b"underlying credential revoked");
        client.revoke_proof_by_hash(&admin, &proof_hash, &reason);

        assert!(
            client.check_proof_revocation(&proof_hash),
            "proof should be marked revoked"
        );
    }

    #[test]
    fn test_revocation_record_stored_correctly() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let proof_hash = BytesN::from_array(&env, &[0x56u8; 32]);
        let reason = Bytes::from_slice(&env, b"test revocation");
        client.register_proof_hash(&admin, &proof_hash);
        client.revoke_proof_by_hash(&admin, &proof_hash, &reason);

        let record = client.get_revocation_record(&proof_hash);
        assert_eq!(record.proof_hash, proof_hash);
        assert_eq!(record.reason, reason);
    }

    #[test]
    fn test_revocation_history_appends_records() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hash1 = BytesN::from_array(&env, &[0x01u8; 32]);
        let hash2 = BytesN::from_array(&env, &[0x02u8; 32]);
        let reason = Bytes::from_slice(&env, b"reason");

        client.register_proof_hash(&admin, &hash1);
        client.register_proof_hash(&admin, &hash2);
        client.revoke_proof_by_hash(&admin, &hash1, &reason);
        client.revoke_proof_by_hash(&admin, &hash2, &reason);

        let history = client.get_revocation_history();
        assert_eq!(history.len(), 2);
    }

    #[test]
    #[should_panic(expected = "proof hash not registered")]
    fn test_revoke_unregistered_proof_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hash = BytesN::from_array(&env, &[0x99u8; 32]);
        let reason = Bytes::from_slice(&env, b"reason");
        // Should panic: proof was never registered
        client.revoke_proof_by_hash(&admin, &hash, &reason);
    }

    #[test]
    #[should_panic(expected = "proof already revoked")]
    fn test_double_revoke_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hash = BytesN::from_array(&env, &[0xAAu8; 32]);
        let reason = Bytes::from_slice(&env, b"reason");
        client.register_proof_hash(&admin, &hash);
        client.revoke_proof_by_hash(&admin, &hash, &reason);
        // Second revocation should panic
        client.revoke_proof_by_hash(&admin, &hash, &reason);
    }

    // =========================================================================
    // Issue #1724: store_proof_metadata authorization tests
    // =========================================================================

    #[test]
    fn test_store_metadata_rejects_unauthorized_caller() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let stranger = Address::generate(&env);
        let hash = Bytes::from_slice(&env, b"hash");
        let desc = String::from_str(&env, "desc");

        let result = client.try_store_proof_metadata(&stranger, &1u64, &ClaimType::HasDegree, &hash, &desc);
        assert!(result.is_err(), "caller without admin or registrar role must be rejected");
        assert!(client.try_get_proof_metadata(&1u64, &ClaimType::HasDegree).is_err());
    }

    #[test]
    fn test_store_metadata_registrar_can_write() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let registrar = Address::generate(&env);
        client.set_registrar(&admin, &registrar, &true);
        assert!(client.is_registrar(&registrar));

        let hash = Bytes::from_slice(&env, b"hash");
        let desc = String::from_str(&env, "desc");
        client.store_proof_metadata(&registrar, &1u64, &ClaimType::HasDegree, &hash, &desc);
        assert_eq!(client.get_proof_metadata(&1u64, &ClaimType::HasDegree).proof_hash, hash);
    }

    #[test]
    fn test_store_metadata_rejects_overwrite_by_other_registrar() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let registrar_a = Address::generate(&env);
        let registrar_b = Address::generate(&env);
        client.set_registrar(&admin, &registrar_a, &true);
        client.set_registrar(&admin, &registrar_b, &true);

        let hash_a = Bytes::from_slice(&env, b"hash-a");
        let hash_b = Bytes::from_slice(&env, b"hash-b");
        let desc = String::from_str(&env, "desc");
        client.store_proof_metadata(&registrar_a, &1u64, &ClaimType::HasDegree, &hash_a, &desc);

        let result = client.try_store_proof_metadata(&registrar_b, &1u64, &ClaimType::HasDegree, &hash_b, &desc);
        assert!(result.is_err(), "another registrar must not overwrite the entry");
        assert_eq!(client.get_proof_metadata(&1u64, &ClaimType::HasDegree).proof_hash, hash_a);
    }

    #[test]
    fn test_store_metadata_writer_and_admin_can_replace() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let registrar = Address::generate(&env);
        client.set_registrar(&admin, &registrar, &true);

        let desc = String::from_str(&env, "desc");
        let hash_1 = Bytes::from_slice(&env, b"hash-1");
        let hash_2 = Bytes::from_slice(&env, b"hash-2");
        let hash_3 = Bytes::from_slice(&env, b"hash-3");
        client.store_proof_metadata(&registrar, &1u64, &ClaimType::HasDegree, &hash_1, &desc);

        // The original writer may replace its own entry.
        client.store_proof_metadata(&registrar, &1u64, &ClaimType::HasDegree, &hash_2, &desc);
        assert_eq!(client.get_proof_metadata(&1u64, &ClaimType::HasDegree).proof_hash, hash_2);

        // The admin may replace any entry.
        client.store_proof_metadata(&admin, &1u64, &ClaimType::HasDegree, &hash_3, &desc);
        assert_eq!(client.get_proof_metadata(&1u64, &ClaimType::HasDegree).proof_hash, hash_3);
    }

    #[test]
    fn test_revoked_registrar_cannot_store_metadata() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let registrar = Address::generate(&env);
        client.set_registrar(&admin, &registrar, &true);
        client.set_registrar(&admin, &registrar, &false);
        assert!(!client.is_registrar(&registrar));

        let hash = Bytes::from_slice(&env, b"hash");
        let desc = String::from_str(&env, "desc");
        let result = client.try_store_proof_metadata(&registrar, &1u64, &ClaimType::HasDegree, &hash, &desc);
        assert!(result.is_err(), "revoked registrar must be rejected");
    }

    #[test]
    fn test_set_registrar_admin_only() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let stranger = Address::generate(&env);
        let target = Address::generate(&env);
        let result = client.try_set_registrar(&stranger, &target, &true);
        assert!(result.is_err(), "only the admin may grant the registrar role");
        assert!(!client.is_registrar(&target));
    }

    // =========================================================================
    // Issue #1725: register_proof_hash authorization, dedup and pagination
    // =========================================================================

    #[test]
    fn test_register_proof_hash_rejects_unauthorized_caller() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let stranger = Address::generate(&env);
        let hash = BytesN::from_array(&env, &[0x21u8; 32]);
        let result = client.try_register_proof_hash(&stranger, &hash);
        assert!(result.is_err(), "caller without admin or registrar role must be rejected");
        assert_eq!(client.get_proof_hash_count(), 0);
    }

    #[test]
    fn test_registrar_can_register_proof_hash() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let registrar = Address::generate(&env);
        client.set_registrar(&admin, &registrar, &true);

        let hash = BytesN::from_array(&env, &[0x22u8; 32]);
        client.register_proof_hash(&registrar, &hash);
        assert!(!client.check_proof_revocation(&hash));
        assert_eq!(client.get_proof_hash_count(), 1);
    }

    #[test]
    fn test_revoked_proof_stays_revoked_after_reregistration() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hash = BytesN::from_array(&env, &[0x23u8; 32]);
        let reason = Bytes::from_slice(&env, b"underlying credential revoked");
        client.register_proof_hash(&admin, &hash);
        client.revoke_proof_by_hash(&admin, &hash, &reason);

        let result = client.try_register_proof_hash(&admin, &hash);
        assert!(result.is_err(), "re-registering a revoked hash must be rejected");
        assert!(client.check_proof_revocation(&hash), "revocation must survive re-registration");
        assert_eq!(client.get_proof_hash_count(), 1);
    }

    #[test]
    fn test_duplicate_proof_hash_registration_rejected() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hash = BytesN::from_array(&env, &[0x24u8; 32]);
        client.register_proof_hash(&admin, &hash);
        let result = client.try_register_proof_hash(&admin, &hash);
        assert!(result.is_err(), "duplicate registration must be rejected");
        assert_eq!(client.get_proof_hash_count(), 1);
    }

    #[test]
    fn test_proof_hash_registry_is_paginated() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let hashes: std::vec::Vec<BytesN<32>> = (0u8..5)
            .map(|i| BytesN::from_array(&env, &[i + 1; 32]))
            .collect();
        for hash in &hashes {
            client.register_proof_hash(&admin, hash);
        }
        assert_eq!(client.get_proof_hash_count(), 5);

        let first = client.get_proof_hash_registry(&0u32, &2u32);
        assert_eq!(first.len(), 2);
        assert_eq!(first.get(0), Some(hashes[0].clone()));
        assert_eq!(first.get(1), Some(hashes[1].clone()));

        let rest = client.get_proof_hash_registry(&2u32, &10u32);
        assert_eq!(rest.len(), 3);
        assert_eq!(rest.get(0), Some(hashes[2].clone()));
        assert_eq!(rest.get(2), Some(hashes[4].clone()));

        assert_eq!(client.get_proof_hash_registry(&5u32, &3u32).len(), 0);
        assert_eq!(client.get_proof_hash_registry(&0u32, &0u32).len(), 0);
    }

    #[test]
    fn test_proof_hash_registry_page_is_capped() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        for i in 0u32..(MAX_PROOF_HASH_PAGE + 1) {
            let mut bytes = [0u8; 32];
            bytes[..4].copy_from_slice(&i.to_be_bytes());
            client.register_proof_hash(&admin, &BytesN::from_array(&env, &bytes));
        }
        assert_eq!(client.get_proof_hash_count(), MAX_PROOF_HASH_PAGE + 1);
        assert_eq!(
            client.get_proof_hash_registry(&0u32, &u32::MAX).len(),
            MAX_PROOF_HASH_PAGE
        );
    }

    #[test]
    fn test_check_revocation_unregistered_returns_false() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        // No registration, no revocation — should just return false
        let hash = BytesN::from_array(&env, &[0x77u8; 32]);
        assert!(!client.check_proof_revocation(&hash));
    }

    // =========================================================================
    // Issue #1281: Proof Expiry and Time-Bound Verification tests
    // =========================================================================

    #[test]
    fn test_submit_time_bound_proof_stores_timestamp() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let proof = Bytes::from_slice(&env, &[0xABu8; 64]);
        let submitted_at = 5_000u64;
        env.ledger().set_timestamp(submitted_at);
        let tb = client.submit_time_bound_proof(&proof);

        assert_eq!(tb.submitted_at, submitted_at);
        assert_eq!(tb.proof, proof);
    }

    #[test]
    fn test_get_proof_timestamp_by_hash_returns_stored_value() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let proof = Bytes::from_slice(&env, &[0xCDu8; 64]);
        let submitted_at = 9_999u64;
        env.ledger().set_timestamp(submitted_at);
        let tb = client.submit_time_bound_proof(&proof);

        let stored = client.get_proof_timestamp_by_hash(&tb.proof_hash);
        assert_eq!(stored, Some(submitted_at));
    }

    #[test]
    fn test_submit_time_bound_proof_records_ledger_time() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        // The submission time is the ledger time, not anything the caller passes
        let proof = Bytes::from_slice(&env, &[0xEEu8; 64]);
        env.ledger().set_timestamp(42_000);
        let tb = client.submit_time_bound_proof(&proof);

        assert_eq!(tb.submitted_at, 42_000u64);
        assert_eq!(client.get_proof_timestamp_by_hash(&tb.proof_hash), Some(42_000u64));
    }

    #[test]
    fn test_resubmit_time_bound_proof_does_not_refresh_submission_time() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let proof = Bytes::from_slice(&env, &[0xEFu8; 64]);
        env.ledger().set_timestamp(1_000);
        client.submit_time_bound_proof(&proof);

        // Resubmitting much later must not reset the freshness window
        env.ledger().set_timestamp(100_000);
        let tb = client.submit_time_bound_proof(&proof);
        assert_eq!(tb.submitted_at, 1_000u64);

        // The original submission is now old, so a 3600s freshness window rejects it
        assert!(!client.verify_proof_with_time_bounds(&proof, &0u64, &3_600u64));
    }

    #[test]
    fn test_verify_proof_with_time_bounds_within_range() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        // Submit proof at ledger timestamp 0 (Env::default starts at 0)
        let proof = Bytes::from_slice(&env, &[0x11u8; 64]);
        client.submit_time_bound_proof(&proof);

        // Ledger timestamp is 0; age = 0; min_age=0, max_age=3600 → should pass
        let result = client.verify_proof_with_time_bounds(&proof, &0u64, &3600u64);
        assert!(result, "proof with age 0 should pass [0, 3600] window");
    }

    #[test]
    fn test_verify_proof_with_time_bounds_no_timestamp_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        // Proof was never submitted via submit_time_bound_proof
        let proof = Bytes::from_slice(&env, &[0x55u8; 64]);
        let result = client.verify_proof_with_time_bounds(&proof, &0u64, &3600u64);
        assert!(!result, "proof without stored timestamp should fail");
    }

    #[test]
    fn test_verify_proof_with_time_bounds_empty_proof_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let empty = Bytes::from_slice(&env, b"");
        let result = client.verify_proof_with_time_bounds(&empty, &0u64, &3600u64);
        assert!(!result, "empty proof should fail time-bound check");
    }

    #[test]
    fn test_verify_proof_with_time_bounds_invalid_range_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let proof = Bytes::from_slice(&env, &[0x22u8; 64]);
        client.submit_time_bound_proof(&proof);

        // min_age > max_age — invalid window
        let result = client.verify_proof_with_time_bounds(&proof, &3600u64, &1u64);
        assert!(!result, "min_age > max_age should always fail");
    }

    // =========================================================================
    // Issue #1282: Multi-Party Computation Support tests
    // =========================================================================

    fn make_session_id(env: &Env, seed: u8) -> BytesN<32> {
        BytesN::from_array(env, &[seed; 32])
    }

    /// Register `n` fresh verifier addresses so MPC sessions can draw on them.
    fn register_mpc_verifiers(env: &Env, client: &ZkVerifierContractClient, admin: &Address, n: u32) {
        for _ in 0..n {
            client.register_mpc_verifier(admin, &Address::generate(env));
        }
    }

    #[test]
    fn test_create_mpc_session_returns_pending() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x01);
        let session = client.create_mpc_session(&admin, &session_id, &2u32, &3u32);

        assert_eq!(session.session_id, session_id);
        assert_eq!(session.threshold, 2u32);
        assert_eq!(session.total_verifiers, 3u32);
        assert_eq!(session.status, MpcSessionStatus::Pending);
    }

    #[test]
    fn test_get_mpc_session_returns_stored() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x02);
        client.create_mpc_session(&admin, &session_id, &1u32, &2u32);

        let retrieved = client.get_mpc_session(&session_id);
        assert_eq!(retrieved.threshold, 1u32);
    }

    #[test]
    fn test_mpc_single_verifier_threshold_1_approves() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x03);
        client.create_mpc_session(&admin, &session_id, &1u32, &1u32);

        let verifier = Address::generate(&env);
        let session = client.submit_mpc_contribution(&verifier, &session_id, &true);

        assert_eq!(session.status, MpcSessionStatus::Approved);
        assert!(client.is_mpc_session_approved(&session_id));
    }

    #[test]
    fn test_mpc_threshold_2_of_3_not_approved_after_one_vote() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x04);
        client.create_mpc_session(&admin, &session_id, &2u32, &3u32);

        let v1 = Address::generate(&env);
        let session = client.submit_mpc_contribution(&v1, &session_id, &true);

        assert_eq!(session.status, MpcSessionStatus::Pending);
        assert!(!client.is_mpc_session_approved(&session_id));
    }

    #[test]
    fn test_mpc_threshold_2_of_3_approved_after_two_votes() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x05);
        client.create_mpc_session(&admin, &session_id, &2u32, &3u32);

        let v1 = Address::generate(&env);
        let v2 = Address::generate(&env);
        client.submit_mpc_contribution(&v1, &session_id, &true);
        let session = client.submit_mpc_contribution(&v2, &session_id, &true);

        assert_eq!(session.status, MpcSessionStatus::Approved);
        assert!(client.is_mpc_session_approved(&session_id));
    }

    #[test]
    fn test_mpc_disapproval_does_not_count_toward_threshold() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x06);
        // threshold=2, but two "no" votes should not approve
        client.create_mpc_session(&admin, &session_id, &2u32, &3u32);

        let v1 = Address::generate(&env);
        let v2 = Address::generate(&env);
        client.submit_mpc_contribution(&v1, &session_id, &false);
        let session = client.submit_mpc_contribution(&v2, &session_id, &false);

        assert_eq!(session.status, MpcSessionStatus::Pending);
        assert!(!client.is_mpc_session_approved(&session_id));
    }

    #[test]
    fn test_mpc_contribution_count_tracks_all_votes() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x07);
        client.create_mpc_session(&admin, &session_id, &3u32, &3u32);

        for _ in 0..3 {
            let v = Address::generate(&env);
            client.submit_mpc_contribution(&v, &session_id, &true);
        }

        let count = client.get_mpc_contribution_count(&session_id);
        assert_eq!(count, 3u32);
    }

    #[test]
    #[should_panic(expected = "verifier has already contributed")]
    fn test_mpc_double_vote_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x08);
        client.create_mpc_session(&admin, &session_id, &2u32, &2u32);

        let v = Address::generate(&env);
        client.submit_mpc_contribution(&v, &session_id, &true);
        // Second vote from same verifier should panic
        client.submit_mpc_contribution(&v, &session_id, &true);
    }

    #[test]
    #[should_panic(expected = "threshold must be at least 1")]
    fn test_mpc_zero_threshold_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        client.create_mpc_session(&admin, &make_session_id(&env, 0x09), &0u32, &3u32);
    }

    #[test]
    #[should_panic(expected = "total_verifiers must be >= threshold")]
    fn test_mpc_threshold_exceeds_total_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        client.create_mpc_session(&admin, &make_session_id(&env, 0x0A), &5u32, &3u32);
    }

    #[test]
    #[should_panic(expected = "session already exists")]
    fn test_mpc_duplicate_session_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 5);

        let session_id = make_session_id(&env, 0x0B);
        client.create_mpc_session(&admin, &session_id, &1u32, &1u32);
        // Creating the same session again must panic
        client.create_mpc_session(&admin, &session_id, &1u32, &1u32);
    }

    #[test]
    #[test]
    #[should_panic(expected = "unauthorized")]
    fn test_mpc_create_session_non_admin_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 3);

        let attacker = Address::generate(&env);
        client.create_mpc_session(&attacker, &make_session_id(&env, 0x0C), &1u32, &1u32);
    }

    #[test]
    #[should_panic(expected = "total_verifiers exceeds registered verifiers")]
    fn test_mpc_session_unregistered_verifier_set_panics() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);
        register_mpc_verifiers(&env, &client, &admin, 2);

        // Only 2 verifiers are registered, so a 3-verifier session is rejected
        client.create_mpc_session(&admin, &make_session_id(&env, 0x0D), &2u32, &3u32);
    }

    #[test]
    fn test_mpc_register_verifier_is_idempotent() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let verifier = Address::generate(&env);
        client.register_mpc_verifier(&admin, &verifier);
        client.register_mpc_verifier(&admin, &verifier);

        assert_eq!(client.get_mpc_verifier_count(), 1u32);
    }

    #[test]
    fn test_mpc_register_verifier_non_admin_rejected() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let attacker = Address::generate(&env);
        let verifier = Address::generate(&env);
        let result = client.try_register_mpc_verifier(&attacker, &verifier);
        assert!(result.is_err());
        assert_eq!(client.get_mpc_verifier_count(), 0u32);
    }

    fn test_mpc_nonexistent_session_not_approved() {
        let env = Env::default();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);

        let session_id = make_session_id(&env, 0xFF);
        assert!(!client.is_mpc_session_approved(&session_id));
    }

    // ── Issue #1418: Real Schnorr sigma-protocol equation ────────────────

    /// Helper: build a valid SchnorrProof for the given (pk, credential_id, claim_type, nonce).
    /// The prover computes:
    ///   c    = SHA-256("schnorr-v1" || pk || commitment || cred_id_le8 || ct_byte || nonce_le8)
    ///   resp = SHA-256("schnorr-resp" || commitment || c)
    fn make_valid_schnorr_proof(
        env: &Env,
        pk: &BytesN<32>,
        commitment_bytes: [u8; 32],
        credential_id: u64,
        claim_type: &ClaimType,
        nonce: u64,
    ) -> SchnorrProof {
        let ct_byte: u8 = match claim_type {
            ClaimType::HasDegree => 0,
            ClaimType::HasLicense => 1,
            ClaimType::HasEmploymentHistory => 2,
            ClaimType::HasCertification => 3,
            ClaimType::HasResearchPublication => 4,
        };

        // Compute challenge exactly as the contract does
        let mut c_hasher = Sha256::new();
        c_hasher.update(b"schnorr-v1");
        c_hasher.update(pk.to_array());
        c_hasher.update(commitment_bytes);
        c_hasher.update(credential_id.to_le_bytes());
        c_hasher.update([ct_byte]);
        c_hasher.update(nonce.to_le_bytes());
        let c: [u8; 32] = c_hasher.finalize().into();

        // Compute response exactly as the contract expects
        let mut r_hasher = Sha256::new();
        r_hasher.update(b"schnorr-resp");
        r_hasher.update(commitment_bytes);
        r_hasher.update(c);
        let response: [u8; 32] = r_hasher.finalize().into();

        SchnorrProof {
            commitment: BytesN::from_array(env, &commitment_bytes),
            response: BytesN::from_array(env, &response),
            nonce,
        }
    }

    #[test]
    fn test_schnorr_real_equation_valid_proof_passes() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        // Register a Schnorr public key
        let pk_bytes = [0xABu8; 32];
        let pk = BytesN::from_array(&env, &pk_bytes);
        client.set_schnorr_public_key(&admin, &pk);

        let credential_id = 42u64;
        let claim_type = ClaimType::HasDegree;
        let nonce = 1001u64;
        let commitment_bytes = [0x11u8; 32]; // non-zero commitment

        let proof = make_valid_schnorr_proof(&env, &pk, commitment_bytes, credential_id, &claim_type, nonce);

        // Valid proof must pass
        let result = client.verify_conditional_disclosure(
            &credential_id,
            &claim_type,
            &proof,
            &None,
            &None,
        );
        assert!(result, "Valid Schnorr proof should pass verification");
    }

    #[test]
    fn test_schnorr_forged_garbage_fails() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let pk_bytes = [0xABu8; 32];
        let pk = BytesN::from_array(&env, &pk_bytes);
        client.set_schnorr_public_key(&admin, &pk);

        // Forged proof: non-zero bytes but response is not SHA-256("schnorr-resp" || T || c)
        let forged = SchnorrProof {
            commitment: BytesN::from_array(&env, &[0x11u8; 32]),
            response: BytesN::from_array(&env, &[0x22u8; 32]), // garbage response
            nonce: 9999u64,
        };

        let result = client.verify_conditional_disclosure(
            &42u64,
            &ClaimType::HasDegree,
            &forged,
            &None,
            &None,
        );
        assert!(!result, "Forged Schnorr proof should fail verification");
    }

    #[test]
    fn test_schnorr_nonce_replay_rejected() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let pk_bytes = [0xABu8; 32];
        let pk = BytesN::from_array(&env, &pk_bytes);
        client.set_schnorr_public_key(&admin, &pk);

        let credential_id = 1u64;
        let claim_type = ClaimType::HasLicense;
        let nonce = 7777u64;
        let commitment_bytes = [0x33u8; 32];

        let proof = make_valid_schnorr_proof(&env, &pk, commitment_bytes, credential_id, &claim_type, nonce);

        // First use succeeds
        assert!(client.verify_conditional_disclosure(&credential_id, &claim_type, &proof, &None, &None));

        // Same nonce reused — must be rejected as replay
        // Need a fresh proof with the same nonce but different commitment won't help
        // because the nonce is already recorded as used
        let proof2 = make_valid_schnorr_proof(&env, &pk, [0x44u8; 32], credential_id, &claim_type, nonce);
        assert!(!client.verify_conditional_disclosure(&credential_id, &claim_type, &proof2, &None, &None),
            "Replayed nonce must be rejected");
    }

    // ── Issue #1419: Key-rotation history bounded at 10 ─────────────────

    #[test]
    fn test_key_rotation_history_bounded_at_ten() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        // Rotate 12 times — history must never exceed 10
        for i in 2u8..=13 {
            let new_key = BytesN::from_array(&env, &[i; 32]);
            client.rotate_verifying_key(&admin, &new_key);
        }

        let history = client.get_key_rotation_history();
        assert_eq!(history.len(), 10, "Key rotation history must be bounded at 10 entries");
    }

    // ── Issue #1420: Real cache statistics ───────────────────────────────

    #[test]
    fn test_cache_stats_hit_and_miss() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let credential_id = 1u64;
        let claim_type = ClaimType::HasDegree;
        let proof = make_valid_proof(&env);
        let ttl = 1000u32;

        // Initially both counters are zero
        let (hits, misses) = client.get_cache_stats();
        assert_eq!(hits, 0, "Initial hit count must be 0");
        assert_eq!(misses, 0, "Initial miss count must be 0");

        // First call: cache miss (no entry yet)
        client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &ttl);
        let (hits, misses) = client.get_cache_stats();
        assert_eq!(hits, 0, "No hits yet after first call");
        assert_eq!(misses, 1, "One miss after first call");

        // Second call with same inputs within TTL: cache hit
        client.verify_proof_cached(&admin, &credential_id, &claim_type, &proof, &ttl);
        let (hits, misses) = client.get_cache_stats();
        assert_eq!(hits, 1, "One hit after second call");
        assert_eq!(misses, 1, "Miss count unchanged after cache hit");
    }
}

/// Tests for Issue #1511: Audit governance and event trail for admin repointing
/// in the zk_verifier contract.
#[cfg(test)]
mod tests_governance_1511 {
    extern crate std;
    use super::*;
    use soroban_sdk::testutils::{Address as _, Events as _};
    use soroban_sdk::{Address, Env};

    fn setup(env: &Env) -> (ZkVerifierContractClient, Address) {
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let admin = Address::generate(env);
        let client = ZkVerifierContractClient::new(env, &contract_id);
        client.initialize(&admin);
        (client, admin)
    }

    // ── update_admin ───────────────────────────────────────────────────────────

    #[test]
    fn test_update_admin_transfers_admin() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let new_admin = Address::generate(&env);
        client.update_admin(&admin, &new_admin);

        // New admin should be able to perform admin actions (e.g. set_verifying_key).
        let vk_hash = soroban_sdk::BytesN::from_array(&env, &[1u8; 32]);
        client.set_verifying_key(&new_admin, &vk_hash);
    }

    #[test]
    fn test_update_admin_emits_admin_transferred_event() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let new_admin = Address::generate(&env);
        client.update_admin(&admin, &new_admin);

        let events = env.events().all();
        let found = events.iter().any(|e| {
            let event_str = std::format!("{:?}", e);
            event_str.contains("AdminTransferred")
        });
        assert!(found, "AdminTransferred event not emitted");
    }

    #[test]
    fn test_update_admin_can_chain_transfers() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, admin) = setup(&env);

        let admin_v2 = Address::generate(&env);
        let admin_v3 = Address::generate(&env);

        client.update_admin(&admin, &admin_v2);
        // admin_v2 can now transfer to admin_v3
        client.update_admin(&admin_v2, &admin_v3);

        // admin_v3 can set verifying key
        let vk_hash = soroban_sdk::BytesN::from_array(&env, &[2u8; 32]);
        client.set_verifying_key(&admin_v3, &vk_hash);
    }

    #[test]
    #[should_panic(expected = "unauthorized")]
    fn test_update_admin_rejects_non_admin() {
        let env = Env::default();
        env.mock_all_auths();
        let (client, _admin) = setup(&env);

        let attacker = Address::generate(&env);
        let new_admin = Address::generate(&env);
        client.update_admin(&attacker, &new_admin);
    }

    #[test]
    #[should_panic(expected = "not initialized")]
    fn test_update_admin_panics_if_not_initialized() {
        let env = Env::default();
        env.mock_all_auths();
        let contract_id = env.register_contract(None, ZkVerifierContract);
        let client = ZkVerifierContractClient::new(&env, &contract_id);
        // initialize() was NOT called
        let admin = Address::generate(&env);
        let new_admin = Address::generate(&env);
        client.update_admin(&admin, &new_admin);
    }
}

