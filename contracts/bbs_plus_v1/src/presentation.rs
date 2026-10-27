
extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::errors::{BbsError, BbsResult};
use crate::primitives::{pairing, Fr, G1, G2};
#[cfg(feature = "std")]
use crate::signature::{compute_b, Signature};
use crate::signature::{base_generator, VerifyingKey};
use crate::transcript::Transcript;
use crate::DOMAIN_BBS_PLUS;

/// Revealed message positions and their values.
pub type RevealedSet = BTreeMap<u32, Fr>;

/// Zero-knowledge proof of possession of a BBS+ signature, selectively
/// disclosing a subset of the signed messages.
///
/// This is a Fiat-Shamir non-interactive adaptation of the BBS proof system
/// from `draft-irtf-cfrg-bbs-signatures` (CoreProofGen/CoreProofVerify),
/// generalized to also cover the signer-chosen `s` blinding scalar: this
/// crate's (A, e, s) signature has no separate deterministic `domain` value
/// the way the IETF draft does, so `s` is treated as an (always-hidden)
/// extra message bound to the verifying key's `q1` generator -- structurally
/// identical to an ordinary undisclosed message, which the proof system
/// already needs to handle, so no separate machinery is needed for it.
#[derive(Clone)]
pub struct PresentationProof {
    /// A_bar = A * (r1*r2): a per-presentation-randomized, unlinkable
    /// re-blinding of the original signature's A component.
    pub a_bar: G1,
    /// B_bar = D*r1 - A_bar*e. Satisfies B_bar = A_bar*SK exactly when the
    /// prover holds a genuine signature -- this is what the final pairing
    /// check verifies, without the verifier ever learning SK, A, or e.
    pub b_bar: G1,
    /// D = B*r2, a blinded commitment to the full message vector.
    pub d: G1,
    /// Schnorr response for e.
    pub e_hat: Fr,
    /// Schnorr response for r1.
    pub r1_hat: Fr,
    /// Schnorr response for r3 = r2^-1.
    pub r3_hat: Fr,
    /// Schnorr response for s.
    pub s_hat: Fr,
    /// Schnorr responses for each hidden (undisclosed) real message, keyed
    /// by message index.
    pub hidden_message_hats: BTreeMap<u32, Fr>,
    /// The disclosed messages, keyed by index.
    pub revealed_messages: RevealedSet,
    /// Fiat-Shamir challenge binding every commitment and disclosed value
    /// above.
    pub challenge: Fr,
}

impl PresentationProof {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&self.a_bar.to_bytes());
        bytes.extend_from_slice(&self.b_bar.to_bytes());
        bytes.extend_from_slice(&self.d.to_bytes());
        bytes.extend_from_slice(&self.e_hat.to_bytes());
        bytes.extend_from_slice(&self.r1_hat.to_bytes());
        bytes.extend_from_slice(&self.r3_hat.to_bytes());
        bytes.extend_from_slice(&self.s_hat.to_bytes());
        bytes.extend_from_slice(&self.challenge.to_bytes());

        bytes.extend_from_slice(&(self.hidden_message_hats.len() as u32).to_le_bytes());
        for (idx, val) in &self.hidden_message_hats {
            bytes.extend_from_slice(&idx.to_le_bytes());
            bytes.extend_from_slice(&val.to_bytes());
        }

        bytes.extend_from_slice(&(self.revealed_messages.len() as u32).to_le_bytes());
        for (idx, val) in &self.revealed_messages {
            bytes.extend_from_slice(&idx.to_le_bytes());
            bytes.extend_from_slice(&val.to_bytes());
        }

        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> BbsResult<Self> {
        const FIXED_LEN: usize = 48 * 3 + 32 * 5;
        if bytes.len() < FIXED_LEN + 8 {
            return Err(BbsError::DeserializationError);
        }
        let mut offset = 0;

        let a_bar = read_g1(bytes, &mut offset)?;
        let b_bar = read_g1(bytes, &mut offset)?;
        let d = read_g1(bytes, &mut offset)?;
        let e_hat = read_fr(bytes, &mut offset)?;
        let r1_hat = read_fr(bytes, &mut offset)?;
        let r3_hat = read_fr(bytes, &mut offset)?;
        let s_hat = read_fr(bytes, &mut offset)?;
        let challenge = read_fr(bytes, &mut offset)?;

        let hidden_message_hats = read_indexed_map(bytes, &mut offset)?;
        let revealed_messages = read_indexed_map(bytes, &mut offset)?;

        if offset != bytes.len() {
            return Err(BbsError::DeserializationError);
        }

        Ok(PresentationProof {
            a_bar,
            b_bar,
            d,
            e_hat,
            r1_hat,
            r3_hat,
            s_hat,
            hidden_message_hats,
            revealed_messages,
            challenge,
        })
    }
}

fn read_g1(bytes: &[u8], offset: &mut usize) -> BbsResult<G1> {
    if *offset + 48 > bytes.len() {
        return Err(BbsError::DeserializationError);
    }
    let mut buf = [0u8; 48];
    buf.copy_from_slice(&bytes[*offset..*offset + 48]);
    *offset += 48;
    G1::from_bytes(&buf)
}

fn read_fr(bytes: &[u8], offset: &mut usize) -> BbsResult<Fr> {
    if *offset + 32 > bytes.len() {
        return Err(BbsError::DeserializationError);
    }
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&bytes[*offset..*offset + 32]);
    *offset += 32;
    Fr::from_bytes(&buf)
}

fn read_indexed_map(bytes: &[u8], offset: &mut usize) -> BbsResult<BTreeMap<u32, Fr>> {
    if *offset + 4 > bytes.len() {
        return Err(BbsError::DeserializationError);
    }
    let mut count_bytes = [0u8; 4];
    count_bytes.copy_from_slice(&bytes[*offset..*offset + 4]);
    let count = u32::from_le_bytes(count_bytes) as usize;
    *offset += 4;

    if count
        .checked_mul(36)
        .map_or(true, |n| offset.saturating_add(n) > bytes.len())
    {
        return Err(BbsError::DeserializationError);
    }

    let mut map = BTreeMap::new();
    for _ in 0..count {
        if *offset + 4 > bytes.len() {
            return Err(BbsError::DeserializationError);
        }
        let mut idx_bytes = [0u8; 4];
        idx_bytes.copy_from_slice(&bytes[*offset..*offset + 4]);
        let idx = u32::from_le_bytes(idx_bytes);
        *offset += 4;
        let val = read_fr(bytes, offset)?;
        if map.insert(idx, val).is_some() {
            return Err(BbsError::DeserializationError);
        }
    }
    Ok(map)
}

/// Random scalars needed by ProofGen. Pulled out so callers control the RNG.
#[cfg(feature = "std")]
struct ProofRandomness {
    r1: Fr,
    r2: Fr,
    e_tilde: Fr,
    r1_tilde: Fr,
    r3_tilde: Fr,
    s_tilde: Fr,
    hidden_tildes: BTreeMap<u32, Fr>,
}

#[cfg(feature = "std")]
fn sample_randomness<R: rand::RngCore>(rng: &mut R, hidden_indices: &[u32]) -> ProofRandomness {
    let nonzero = |rng: &mut R| loop {
        let f = Fr::random(rng);
        if !f.is_zero() {
            return f;
        }
    };
    let mut hidden_tildes = BTreeMap::new();
    for &idx in hidden_indices {
        hidden_tildes.insert(idx, Fr::random(rng));
    }
    ProofRandomness {
        r1: nonzero(rng),
        r2: nonzero(rng),
        e_tilde: Fr::random(rng),
        r1_tilde: Fr::random(rng),
        r3_tilde: Fr::random(rng),
        s_tilde: Fr::random(rng),
        hidden_tildes,
    }
}

fn compute_challenge(
    vk: &VerifyingKey,
    revealed: &RevealedSet,
    a_bar: &G1,
    b_bar: &G1,
    d: &G1,
    t1: &G1,
    t2: &G1,
    presentation_context: &[u8],
) -> BbsResult<Fr> {
    let mut transcript = Transcript::new(DOMAIN_BBS_PLUS.as_bytes());
    transcript.append(b"vk_w", &vk.w.to_bytes());
    transcript.append_g1(b"a_bar", &a_bar.to_bytes());
    transcript.append_g1(b"b_bar", &b_bar.to_bytes());
    transcript.append_g1(b"d", &d.to_bytes());
    transcript.append_g1(b"t1", &t1.to_bytes());
    transcript.append_g1(b"t2", &t2.to_bytes());
    for (idx, val) in revealed {
        transcript.append(b"rev_idx", &idx.to_le_bytes());
        transcript.append_scalar(b"rev_val", val);
    }
    transcript.append(b"ctx", presentation_context);
    transcript.squeeze_challenge()
}

/// BBS+ Presentation Proof System.
pub struct BbsPresentation;

impl BbsPresentation {
    /// Create a presentation proof for selective disclosure.
    ///
    /// `messages` must be the full signed message vector (in the order they
    /// were signed); `revealed_indices` selects which of those to disclose
    /// in the clear. `presentation_context` is caller-supplied and MUST be
    /// fresh (e.g. a verifier-issued nonce) -- it's what makes replaying a
    /// captured proof against a different verifier/session fail.
    #[cfg(feature = "std")]
    pub fn create_presentation<R: rand::RngCore>(
        rng: &mut R,
        signature: &Signature,
        vk: &VerifyingKey,
        messages: &[Fr],
        revealed_indices: &[u32],
        presentation_context: &[u8],
    ) -> BbsResult<PresentationProof> {
        if messages.len() != vk.message_generators.len() {
            return Err(BbsError::InvalidMessageCount);
        }
        for &idx in revealed_indices {
            if idx as usize >= messages.len() {
                return Err(BbsError::InvalidProofStructure);
            }
        }

        let l = messages.len() as u32;
        let hidden_indices: Vec<u32> = (0..l).filter(|i| !revealed_indices.contains(i)).collect();

        let rnd = sample_randomness(rng, &hidden_indices);

        let b = compute_b(vk, messages, &signature.s)?;
        let r3 = rnd.r2.invert()?;

        let d = b.mul(&rnd.r2);
        let a_bar = signature.a.mul(&rnd.r1.mul(&rnd.r2));
        let b_bar = d.mul(&rnd.r1).sub(&a_bar.mul(&signature.e));

        // T1 = Abar*e~ + D*r1~
        let t1 = a_bar.mul(&rnd.e_tilde).add(&d.mul(&rnd.r1_tilde));

        // T2 = D*r3~ + Q1*s~ + sum(H_j * m~_j) over hidden j
        let mut t2 = d.mul(&rnd.r3_tilde).add(&vk.q1.mul(&rnd.s_tilde));
        for &idx in &hidden_indices {
            let h_j = &vk.message_generators[idx as usize];
            let m_tilde_j = rnd.hidden_tildes.get(&idx).expect("sampled above");
            t2 = t2.add(&h_j.mul(m_tilde_j));
        }

        let mut revealed_messages = RevealedSet::new();
        for &idx in revealed_indices {
            revealed_messages.insert(idx, messages[idx as usize]);
        }

        let challenge = compute_challenge(
            vk,
            &revealed_messages,
            &a_bar,
            &b_bar,
            &d,
            &t1,
            &t2,
            presentation_context,
        )?;

        let e_hat = rnd.e_tilde.add(&signature.e.mul(&challenge));
        let r1_hat = rnd.r1_tilde.sub(&rnd.r1.mul(&challenge));
        let r3_hat = rnd.r3_tilde.sub(&r3.mul(&challenge));
        let s_hat = rnd.s_tilde.add(&signature.s.mul(&challenge));

        let mut hidden_message_hats = BTreeMap::new();
        for &idx in &hidden_indices {
            let m_tilde_j = rnd.hidden_tildes.get(&idx).expect("sampled above");
            let m_hat_j = m_tilde_j.add(&messages[idx as usize].mul(&challenge));
            hidden_message_hats.insert(idx, m_hat_j);
        }

        Ok(PresentationProof {
            a_bar,
            b_bar,
            d,
            e_hat,
            r1_hat,
            r3_hat,
            s_hat,
            hidden_message_hats,
            revealed_messages,
            challenge,
        })
    }

    /// Verify a presentation proof against a verifying key and the
    /// `presentation_context` the verifier itself issued/expects.
    pub fn verify_presentation(
        vk: &VerifyingKey,
        proof: &PresentationProof,
        presentation_context: &[u8],
    ) -> BbsResult<bool> {
        let l = vk.message_generators.len() as u32;
        for &idx in proof.revealed_messages.keys() {
            if idx >= l {
                return Err(BbsError::InvalidProofStructure);
            }
        }
        for &idx in proof.hidden_message_hats.keys() {
            if idx >= l {
                return Err(BbsError::InvalidProofStructure);
            }
        }
        // Every message index must appear in exactly one of the two sets.
        if (proof.revealed_messages.len() + proof.hidden_message_hats.len()) as u32 != l {
            return Err(BbsError::InvalidProofStructure);
        }
        for idx in proof.revealed_messages.keys() {
            if proof.hidden_message_hats.contains_key(idx) {
                return Err(BbsError::InvalidProofStructure);
            }
        }

        if proof.a_bar.is_identity() {
            return Ok(false);
        }

        // T1' = Bbar*c + Abar*e^ + D*r1^
        let t1 = proof
            .b_bar
            .mul(&proof.challenge)
            .add(&proof.a_bar.mul(&proof.e_hat))
            .add(&proof.d.mul(&proof.r1_hat));

        // Bv = P1 + sum(H_i * revealed_i)
        let mut bv = base_generator();
        for (&idx, val) in &proof.revealed_messages {
            bv = bv.add(&vk.message_generators[idx as usize].mul(val));
        }

        // T2' = Bv*c + D*r3^ + Q1*s^ + sum(H_j * m^_j) over hidden j
        let mut t2 = bv
            .mul(&proof.challenge)
            .add(&proof.d.mul(&proof.r3_hat))
            .add(&vk.q1.mul(&proof.s_hat));
        for (&idx, m_hat) in &proof.hidden_message_hats {
            t2 = t2.add(&vk.message_generators[idx as usize].mul(m_hat));
        }

        let expected_challenge = compute_challenge(
            vk,
            &proof.revealed_messages,
            &proof.a_bar,
            &proof.b_bar,
            &proof.d,
            &t1,
            &t2,
            presentation_context,
        )?;

        if proof.challenge != expected_challenge {
            return Ok(false);
        }

        // e(Abar, W) == e(Bbar, BP2)  <=>  Bbar == Abar*SK, which only holds
        // when Abar was derived from a genuine signature (see derivation in
        // presentation.rs module docs / commit message).
        let lhs = pairing(&proof.a_bar, &vk.w);
        let rhs = pairing(&proof.b_bar, &G2::generator());

        Ok(lhs == rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::BbsSignature;
    use alloc::vec;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    fn setup(n: u32) -> (crate::SigningKey, VerifyingKey) {
        let mut r = rng();
        let sk = crate::SigningKey::generate(&mut r);
        let vk = VerifyingKey::derive(sk.public_key(), b"presentation-test-ctx", n).unwrap();
        (sk, vk)
    }

    #[test]
    fn test_presentation_proof_serialization_roundtrip() {
        let (sk, vk) = setup(3);
        let mut r = rng();
        let messages = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let proof =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[0], b"nonce-1")
                .unwrap();

        let bytes = proof.to_bytes();
        let proof2 = PresentationProof::from_bytes(&bytes).unwrap();

        assert_eq!(proof.challenge, proof2.challenge);
        assert_eq!(proof.a_bar, proof2.a_bar);
        assert_eq!(proof.b_bar, proof2.b_bar);
        assert_eq!(proof.d, proof2.d);
        assert_eq!(proof.revealed_messages, proof2.revealed_messages);
        assert_eq!(proof.hidden_message_hats, proof2.hidden_message_hats);
    }

    #[test]
    fn test_create_and_verify_presentation_reveal_one() {
        let (sk, vk) = setup(2);
        let mut r = rng();
        let messages = vec![Fr::from_u64(11), Fr::from_u64(22)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let proof =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[0], b"nonce")
                .unwrap();
        let valid = BbsPresentation::verify_presentation(&vk, &proof, b"nonce").unwrap();
        assert!(valid);
        assert_eq!(proof.revealed_messages.get(&0), Some(&Fr::from_u64(11)));
        assert!(proof.revealed_messages.get(&1).is_none());
    }

    #[test]
    fn test_create_and_verify_presentation_reveal_none() {
        let (sk, vk) = setup(2);
        let mut r = rng();
        let messages = vec![Fr::from_u64(11), Fr::from_u64(22)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let proof =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[], b"nonce")
                .unwrap();
        assert!(BbsPresentation::verify_presentation(&vk, &proof, b"nonce").unwrap());
        assert!(proof.revealed_messages.is_empty());
    }

    #[test]
    fn test_create_and_verify_presentation_reveal_all() {
        let (sk, vk) = setup(2);
        let mut r = rng();
        let messages = vec![Fr::from_u64(11), Fr::from_u64(22)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let proof =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[0, 1], b"nonce")
                .unwrap();
        assert!(BbsPresentation::verify_presentation(&vk, &proof, b"nonce").unwrap());
    }

    #[test]
    fn test_wrong_context_fails_verification() {
        let (sk, vk) = setup(2);
        let mut r = rng();
        let messages = vec![Fr::from_u64(11), Fr::from_u64(22)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let proof = BbsPresentation::create_presentation(
            &mut r,
            &sig,
            &vk,
            &messages,
            &[0],
            b"expected-nonce",
        )
        .unwrap();

        let valid =
            BbsPresentation::verify_presentation(&vk, &proof, b"different-nonce").unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_tampered_revealed_message_fails_verification() {
        let (sk, vk) = setup(2);
        let mut r = rng();
        let messages = vec![Fr::from_u64(11), Fr::from_u64(22)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let mut proof =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[0], b"nonce")
                .unwrap();
        proof.revealed_messages.insert(0, Fr::from_u64(999));

        let valid = BbsPresentation::verify_presentation(&vk, &proof, b"nonce").unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_proof_without_a_genuine_signature_is_rejected() {
        // The regression test for the old placeholder implementation: a
        // "proof" built from unrelated points (not derived from an actual
        // valid signature via r1/r2 randomization) must fail the final
        // pairing check, even though transcript/challenge bookkeeping alone
        // might look internally consistent.
        let (_, vk) = setup(1);
        let forged = PresentationProof {
            a_bar: G1::generator(),
            b_bar: G1::generator(),
            d: G1::generator(),
            e_hat: Fr::one(),
            r1_hat: Fr::one(),
            r3_hat: Fr::one(),
            s_hat: Fr::one(),
            hidden_message_hats: BTreeMap::new(),
            revealed_messages: {
                let mut m = RevealedSet::new();
                m.insert(0, Fr::one());
                m
            },
            challenge: Fr::one(),
        };
        // Either the challenge recomputation fails to match (most likely)
        // or, in the vanishingly unlikely case it did match, the pairing
        // check must still catch it -- both paths return `Ok(false)`, never
        // `Ok(true)`.
        assert!(!BbsPresentation::verify_presentation(&vk, &forged, b"nonce").unwrap());
    }

    #[test]
    fn test_two_presentations_of_same_signature_are_unlinkable() {
        // A_bar, B_bar, D are independently randomized by fresh r1, r2 each
        // call, so two presentations of the *same* underlying signature
        // must not share any of their EC commitments.
        let (sk, vk) = setup(1);
        let mut r = rng();
        let messages = vec![Fr::from_u64(5)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let proof1 =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[], b"ctx-a")
                .unwrap();
        let proof2 =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[], b"ctx-b")
                .unwrap();

        assert_ne!(proof1.a_bar, proof2.a_bar);
        assert_ne!(proof1.b_bar, proof2.b_bar);
        assert_ne!(proof1.d, proof2.d);
    }

    #[test]
    fn test_reveal_index_out_of_range_rejected() {
        let (sk, vk) = setup(2);
        let mut r = rng();
        let messages = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();

        let result =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[5], b"nonce");
        assert!(result.is_err());
    }

    #[test]
    fn test_presentation_proof_from_bytes_rejects_truncated_and_corrupt_inputs() {
        let (sk, vk) = setup(2);
        let mut r = rng();
        let messages = vec![Fr::from_u64(7), Fr::from_u64(13)];
        let sig = BbsSignature::sign(&mut r, &sk, &vk, &messages).unwrap();
        let proof =
            BbsPresentation::create_presentation(&mut r, &sig, &vk, &messages, &[0], b"nonce")
                .unwrap();
        let valid_bytes = proof.to_bytes();

        // Any prefix shorter than minimum 312 bytes (including 272..311 range) must fail cleanly
        for len in [0usize, 1, 272, 280, 304, 311] {
            assert!(matches!(
                PresentationProof::from_bytes(&valid_bytes[..len]),
                Err(BbsError::DeserializationError)
            ));
        }

        // Trailing bytes must be rejected
        let mut with_trailing = valid_bytes.clone();
        with_trailing.push(0xAA);
        assert!(matches!(
            PresentationProof::from_bytes(&with_trailing),
            Err(BbsError::DeserializationError)
        ));

        // Oversized map count in header must be rejected immediately without OOM
        let mut oversized_count = valid_bytes[..312].to_vec();
        oversized_count[304..308].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            PresentationProof::from_bytes(&oversized_count),
            Err(BbsError::DeserializationError)
        ));
    }
}
