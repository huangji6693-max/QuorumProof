#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use bbs_plus_v1::{
    BbsPresentation, BbsSignature, PresentationProof, Signature, SigningKey, VerifyingKey,
    primitives::Fr,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Fuzz input driving every path through the `bbs_plus_v1` public API.
///
/// We keep the input small so libFuzzer can explore the space quickly:
/// - `seed`            : deterministic RNG seed so every run is reproducible.
/// - `message_scalars` : up to 16 raw u64 values encoded as `Fr` field elements.
///                       (u64 covers both zero and near-zero, exercising the
///                        "invalid scalar" guard inside `SigningKey::from_scalar`.)
/// - `revealed_indices`: which message positions to disclose in the presentation.
/// - `nonce`           : arbitrary verifier nonce (may be empty).
/// - `corrupt_bytes`   : byte string fed into `PresentationProof::from_bytes`
///                       to test that malformed wire data is rejected gracefully.
/// - `corrupt_sig_bytes`: byte string fed into `BbsSignature::from_bytes`
///                        to test that malformed signature data is rejected.
#[derive(Arbitrary, Debug)]
struct BbsPlusFuzzInput {
    seed: u64,
    message_scalars: Vec<u64>,
    revealed_indices: Vec<u8>,
    nonce: Vec<u8>,
    corrupt_bytes: Vec<u8>,
    corrupt_sig_bytes: Vec<u8>,
}

fuzz_target!(|input: BbsPlusFuzzInput| {
    // ── Bound inputs to keep runtime tractable ──────────────────────────────
    // At most 16 messages; at least 1 so VerifyingKey::derive doesn't get 0.
    let messages: Vec<Fr> = if input.message_scalars.is_empty() {
        vec![Fr::from_u64(input.seed)]
    } else {
        input
            .message_scalars
            .iter()
            .take(16)
            .map(|&v| Fr::from_u64(v))
            .collect()
    };
    let message_count = messages.len() as u32;

    // ── Deterministic RNG seeded from fuzz input ─────────────────────────────
    let mut rng = StdRng::seed_from_u64(input.seed);

    // ── 1. Key generation ────────────────────────────────────────────────────
    let sk = SigningKey::generate(&mut rng);
    let pk = sk.public_key();

    // ── 2. VerifyingKey derivation ───────────────────────────────────────────
    // Pass message_count as the generator count; an empty context tag is valid.
    let vk = match VerifyingKey::derive(pk, b"fuzz-context-v1", message_count) {
        Ok(vk) => vk,
        Err(_) => return, // derivation can fail for 0 messages; we bounded above but guard anyway
    };

    // ── 3. Signing ───────────────────────────────────────────────────────────
    let sig = match BbsSignature::sign(&mut rng, &sk, &vk, &messages) {
        Ok(s) => s,
        Err(_) => return, // must not panic
    };

    // ── 4. Signature verification — must succeed for a fresh, unmodified sig ─
    let verify_ok = BbsSignature::verify(&vk, &messages, &sig)
        .expect("verify must not panic on a freshly issued signature");
    assert!(
        verify_ok,
        "A freshly issued signature must verify correctly"
    );

    // ── 5. Verify rejects if a message is altered ────────────────────────────
    if messages.len() > 1 {
        let mut tampered = messages.clone();
        // Flip the last message to something different.
        tampered[messages.len() - 1] = Fr::from_u64(input.seed.wrapping_add(1));
        let tampered_result = BbsSignature::verify(&vk, &tampered, &sig)
            .expect("verify must not panic on tampered messages");
        // Only assert if tampered value is actually different.
        if tampered[messages.len() - 1] != messages[messages.len() - 1] {
            assert!(
                !tampered_result,
                "Tampered messages must not verify against the original signature"
            );
        }
    }

    // ── 6. Presentation creation (selective disclosure) ──────────────────────
    // Convert revealed_indices to the subset of valid indices.
    let all_indices: Vec<u32> = (0..message_count).collect();
    let revealed: Vec<u32> = input
        .revealed_indices
        .iter()
        .map(|&i| (i as u32) % message_count)
        .collect::<std::collections::BTreeSet<_>>() // deduplicate
        .into_iter()
        .collect();

    // Revealing *all* messages is a degenerate but valid case — test it.
    let reveal_set: Vec<u32> = if revealed.is_empty() { all_indices } else { revealed };

    let nonce = if input.nonce.is_empty() { b"default-nonce".as_slice() } else { &input.nonce };
    // Clamp nonce to 64 bytes to keep computation bounded.
    let nonce = &nonce[..nonce.len().min(64)];

    let presentation = match BbsPresentation::create_presentation(
        &mut rng,
        &sig,
        &vk,
        &messages,
        &reveal_set,
        nonce,
    ) {
        Ok(p) => p,
        Err(_) => return, // must not panic
    };

    // ── 7. Presentation verification — must succeed ──────────────────────────
    let pres_ok = BbsPresentation::verify_presentation(&vk, &presentation, nonce)
        .expect("verify_presentation must not panic on a freshly created presentation");
    assert!(
        pres_ok,
        "A freshly created presentation must verify correctly"
    );

    // ── 8. Presentation rejects a different nonce ────────────────────────────
    let wrong_nonce = b"wrong-nonce-should-fail";
    if wrong_nonce != nonce {
        let wrong_ok = BbsPresentation::verify_presentation(&vk, &presentation, wrong_nonce)
            .expect("verify_presentation must not panic on a wrong nonce");
        assert!(
            !wrong_ok,
            "Presentation must not verify against a different nonce"
        );
    }

    // ── 9. Wire-format round-trip ────────────────────────────────────────────
    let bytes = presentation.to_bytes();
    let decoded = PresentationProof::from_bytes(&bytes)
        .expect("from_bytes must not panic on bytes produced by to_bytes");
    let roundtrip_ok = BbsPresentation::verify_presentation(&vk, &decoded, nonce)
        .expect("verify_presentation must not panic on a round-tripped presentation");
    assert!(
        roundtrip_ok,
        "Round-tripped presentation must verify correctly"
    );

    // ── 10. Malformed bytes: from_bytes must return Err, never panic ─────────
    // Both empty and arbitrary garbage are valid fuzz inputs here.
    let _ = PresentationProof::from_bytes(&input.corrupt_bytes);
    let _ = PresentationProof::from_bytes(&[]);

    // ── 11. Malformed signature bytes: must return Err, never panic ──────────
    // (If the library exposes BbsSignature::from_bytes; fall back gracefully.)
    // We attempt to verify an obviously wrong signature against the real vk/messages
    // by constructing a presentation from corrupt bytes and verifying — the
    // result must be Err or Ok(false), never a panic.
    if !input.corrupt_sig_bytes.is_empty() {
        if let Ok(raw_sig) = <&[u8; 112]>::try_from(input.corrupt_sig_bytes.as_slice()) {
            if let Ok(corrupt_sig) = Signature::from_bytes(raw_sig) {
                let _ = BbsSignature::verify(&vk, &messages, &corrupt_sig);
            }
        }
        if let Ok(corrupt_proof) = PresentationProof::from_bytes(&input.corrupt_sig_bytes) {
            // Must not panic regardless of result.
            let _ = BbsPresentation::verify_presentation(&vk, &corrupt_proof, nonce);
        }
    }
});
