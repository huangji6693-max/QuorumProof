#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use soroban_sdk::{testutils::Address as _, Address, Bytes, Env, Vec};
use quorum_proof::{QuorumProofContract, QuorumProofContractClient};

/// Fuzz testing for batch credential issuance operations at scale
/// 
/// This fuzz target focuses on:
/// - Batch issuance of 1000+ credentials with random parameters
/// - Performance testing with various batch configurations
/// - Memory and storage efficiency under stress
/// - Identifying bottlenecks in batch operations
/// - Verifying data integrity across large batches
#[derive(Arbitrary, Debug)]
struct BatchOperationsFuzzInput {
    // Batch size variations (10-2000 credentials)
    batch_size: u16,
    
    // Number of sequential batches to issue
    batch_iterations: u8,
    
    // Metadata complexity (affects per-credential processing)
    metadata_complexity: u8,
    
    // Credential type seed
    type_seed: u32,
    
    // Subject address variation pattern
    subject_pattern: u8,
    
    // Test configuration flags
    test_single_issuer: bool,
    test_multiple_issuers: bool,
    test_verification: bool,
}

fuzz_target!(|input: BatchOperationsFuzzInput| {
    let env = Env::default();
    env.mock_all_auths();

    let contract_id = env.register_contract(None, QuorumProofContract);
    let client = QuorumProofContractClient::new(&env, &contract_id);
    let admin = Address::generate(&env);
    
    // Initialize contract
    if let Err(_) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        client.initialize(&admin)
    })) {
        return;
    }

    // Test 1: Single issuer batch operations
    if input.test_single_issuer {
        test_single_issuer_batch(&env, &client, &input);
    }

    // Test 2: Multiple issuers concurrent batches
    if input.test_multiple_issuers {
        test_multiple_issuers_batch(&env, &client, &input);
    }

    // Test 3: Large batch with integrity verification
    if input.test_verification {
        test_large_batch_with_verification(&env, &client, &input);
    }

    // Test 4: Stress test with maximum batch size
    test_maximum_batch_size(&env, &client, &input);
});

/// Test single issuer issuing large batch of credentials
fn test_single_issuer_batch(env: &Env, client: &QuorumProofContractClient, input: &BatchOperationsFuzzInput) {
    let issuer = Address::generate(&env);
    let batch_size = (input.batch_size as usize).clamp(10, 1000);

    let mut issued_count = 0u64;
    let mut failed_count = 0u64;

    for i in 0..batch_size {
        let subject = match input.subject_pattern {
            0 => Address::generate(&env),
            1 => {
                // Deterministic subject based on index
                let mut addr_seed = [0u8; 32];
                for j in 0..4 {
                    addr_seed[j] = ((i >> (j * 8)) & 0xFF) as u8;
                }
                Address::generate(&env) // Generate but use different for each
            }
            _ => Address::generate(&env),
        };

        let credential_type = 1 + (input.type_seed.wrapping_add(i as u32) % (u32::MAX - 1));
        let metadata = generate_metadata_for_batch(input.metadata_complexity, i as u32);
        let meta_bytes = Bytes::from_slice(&env, &metadata);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            client.issue_credential(&issuer, &subject, &credential_type, &meta_bytes, &None, &0u64)
        }));

        match result {
            Ok(_) => issued_count += 1,
            Err(_) => failed_count += 1,
        }

        // Track failures but continue testing
        if failed_count > (batch_size as u64 / 2) {
            // If failure rate exceeds 50%, stop to avoid infinite failures
            break;
        }
    }

    // Verify batch was created successfully (at least 80% success rate)
    let success_rate = if batch_size > 0 {
        (issued_count as f64) / (batch_size as f64)
    } else {
        0.0
    };

    assert!(success_rate >= 0.8, "Batch issuance success rate too low: {}", success_rate);
}

/// Test multiple issuers issuing batches concurrently
fn test_multiple_issuers_batch(env: &Env, client: &QuorumProofContractClient, input: &BatchOperationsFuzzInput) {
    let num_issuers = ((input.batch_iterations as usize) % 5).max(2); // 2-5 issuers
    let batch_size = ((input.batch_size as usize) / num_issuers).clamp(10, 500);

    let mut all_credentials = vec![];

    for issuer_idx in 0..num_issuers {
        let issuer = Address::generate(&env);

        for cred_idx in 0..batch_size {
            let subject = Address::generate(&env);
            let credential_type =
                1 + (input.type_seed.wrapping_add((issuer_idx as u32 * 10000) + cred_idx as u32) % (u32::MAX - 1));
            let metadata = generate_metadata_for_batch(input.metadata_complexity, (issuer_idx as u32 * 1000) + cred_idx as u32);
            let meta_bytes = Bytes::from_slice(&env, &metadata);

            if let Ok(cred_id) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                client.issue_credential(&issuer, &subject, &credential_type, &meta_bytes, &None, &0u64)
            })) {
                all_credentials.push((issuer.clone(), cred_id));
            }
        }
    }

    // Verify credentials from different issuers were created
    assert!(all_credentials.len() > 0, "No credentials created in multi-issuer batch");
}

/// Test large batch with integrity verification
fn test_large_batch_with_verification(env: &Env, client: &QuorumProofContractClient, input: &BatchOperationsFuzzInput) {
    let mut issuer = Address::generate(&env);
    let batch_size = (input.batch_size as usize).clamp(100, 2000); // 100-2000 for scale testing

    let mut credential_ids = vec![];
    let mut issuers = vec![];
    let mut subjects = vec![];

    // Issue batch (rotating issuer every 900 calls so per-issuer rate limit of 1000 does not reject valid batch items)
    for i in 0..batch_size {
        if i > 0 && i % 900 == 0 {
            issuer = Address::generate(&env);
        }
        let subject = Address::generate(&env);
        let credential_type = 1 + (2000u32.wrapping_add(i as u32) % (u32::MAX - 1));
        let metadata = generate_metadata_for_batch(input.metadata_complexity, i as u32);
        let meta_bytes = Bytes::from_slice(&env, &metadata);

        if let Ok(cred_id) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            client.issue_credential(&issuer, &subject, &credential_type, &meta_bytes, &None, &0u64)
        })) {
            credential_ids.push(cred_id);
            issuers.push(issuer.clone());
            subjects.push(subject);
        }
    }

    // Verify integrity: sample verification across the batch
    let sample_rate = if credential_ids.len() > 100 { 10 } else { 2 };
    for (idx, &cred_id) in credential_ids.iter().enumerate().step_by(sample_rate.max(1)) {
        if let Ok(credential) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            client.get_credential(&cred_id)
        })) {
            // Verify critical properties
            assert_eq!(credential.issuer, issuers[idx], "Issuer mismatch at credential {}", idx);
            assert_eq!(credential.subject, subjects[idx], "Subject mismatch at credential {}", idx);
            assert!(!credential.revoked, "Credential {} should not be revoked", idx);
            assert!(credential.credential_type > 0, "Invalid credential type at {}", idx);
        }
    }
}

/// Test maximum batch size that contract can handle
fn test_maximum_batch_size(env: &Env, client: &QuorumProofContractClient, input: &BatchOperationsFuzzInput) {
    // Test progressively larger batches to find limits
    let test_sizes = vec![100, 500, 1000, 2000];
    
    for &size in &test_sizes {
        let mut issuer = Address::generate(&env);
        let mut successful = 0u64;
        let mut failed = 0u64;

        for i in 0..size {
            if i > 0 && i % 900 == 0 {
                issuer = Address::generate(&env);
            }
            let subject = Address::generate(&env);
            let credential_type = 1 + (input.type_seed.wrapping_add(i as u32) % (u32::MAX - 1));
            let metadata = generate_metadata_for_batch(input.metadata_complexity, i as u32);
            let meta_bytes = Bytes::from_slice(&env, &metadata);

            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                client.issue_credential(&issuer, &subject, &credential_type, &meta_bytes, &None, &0u64)
            })) {
                Ok(_) => successful += 1,
                Err(_) => {
                    failed += 1;
                    // Stop if too many failures
                    if failed > (size as u64 / 2) {
                        break;
                    }
                }
            }
        }

        // Ensure at least 50% success rate for batch testing
        let success_rate = (successful as f64) / ((successful + failed) as f64).max(1.0);
        assert!(success_rate >= 0.5, "Batch size {} has insufficient success rate: {}", size, success_rate);
    }
}

/// Generate metadata with varying complexity
fn generate_metadata_for_batch(complexity: u8, seed: u32) -> Vec<u8> {
    let size = match complexity % 4 {
        0 => 32,      // Small metadata
        1 => 96,      // Medium metadata
        2 => 256,     // Large metadata (max)
        _ => 64,      // Standard metadata
    };

    let mut metadata = vec![0u8; size];
    let mut rng = seed;

    // Use LCG for pseudo-random generation
    for i in 0..size {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        metadata[i] = ((rng >> 16) ^ (rng >> 8)) as u8;
    }

    metadata
}
