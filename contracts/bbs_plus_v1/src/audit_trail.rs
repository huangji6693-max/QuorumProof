/// # BBS+ Disclosure Audit Trail — Issue #1292
///
/// Every time a selective-disclosure proof is created, a log entry is
/// appended recording *what* was disclosed (the revealed field indices), *to
/// whom* (verifier identifier — may be a DID, public key hash, or session ID
/// the caller supplies), and *when* (a caller-supplied UNIX timestamp).
///
/// ## Design
/// - `DisclosureRecord` captures a single proof-creation event.
/// - `AuditTrail` is the in-process log store (BTreeMap keyed on
///   credential_id → Vec<DisclosureRecord>).
/// - `get_disclosure_audit_trail` returns the full history for a credential.
/// - `query` supports optional filtering by date range and/or disclosed fields.
/// - The trail is append-only; records are never mutated after insertion.
///
/// ## Privacy
/// Verifier identifiers and timestamps are caller-supplied and are never
/// stored on-chain by this module.  Callers that want on-chain accountability
/// should emit Soroban events or store a hash; this module provides the
/// in-process accumulation point for off-chain or ledger-backed transports.
///
/// ## Issue requirements mapping
/// - "Log all disclosure proof creations with timestamp, disclosed fields,
///   and target" → `log_disclosure`
/// - "Implement `get_disclosure_audit_trail(env, credential_id) -> Vec`" →
///   `get_disclosure_audit_trail`
/// - "Add query filtering by date range and disclosed fields" →
///   `AuditTrail::query` / `AuditQuery`
/// - "Add tests for audit trail completeness" → test module below

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// A single audit record for one selective-disclosure proof creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisclosureRecord {
    /// Credential identifier this record belongs to (caller-defined bytes,
    /// e.g. a Soroban u64 serialized as little-endian or a DID fragment).
    pub credential_id: Vec<u8>,

    /// UNIX timestamp (seconds since epoch) at the time of disclosure.
    /// The caller is responsible for providing this — the library is no_std
    /// and has no access to a system clock.
    pub timestamp: u64,

    /// Sorted, deduplicated set of revealed message indices.
    pub disclosed_fields: Vec<u32>,

    /// Optional identifier for the verifier / relying party.  May be a DID,
    /// a hashed public key, a session token, etc.  `None` if the disclosure
    /// was made without a specific target (e.g. a public credential).
    pub target: Option<Vec<u8>>,

    /// Optional human-readable label for each disclosed field (index →
    /// label string).  Helps auditors read the trail without the schema.
    pub field_labels: BTreeMap<u32, Vec<u8>>,
}

impl DisclosureRecord {
    /// Construct a minimal disclosure record.
    pub fn new(
        credential_id: &[u8],
        timestamp: u64,
        disclosed_fields: &[u32],
        target: Option<&[u8]>,
    ) -> Self {
        let mut fields: Vec<u32> = disclosed_fields.to_vec();
        fields.sort_unstable();
        fields.dedup();

        DisclosureRecord {
            credential_id: credential_id.to_vec(),
            timestamp,
            disclosed_fields: fields,
            target: target.map(|t| t.to_vec()),
            field_labels: BTreeMap::new(),
        }
    }

    /// Attach a human-readable label to a disclosed field index.
    pub fn with_label(mut self, index: u32, label: &[u8]) -> Self {
        self.field_labels.insert(index, label.to_vec());
        self
    }

    /// True if `field_index` appears in this record's disclosed set.
    pub fn discloses(&self, field_index: u32) -> bool {
        self.disclosed_fields.binary_search(&field_index).is_ok()
    }
}

/// Filter parameters for `AuditTrail::query`.
///
/// All fields are optional; omitting a filter means "no restriction on this
/// dimension".
#[derive(Clone, Debug, Default)]
pub struct AuditQuery {
    /// Lower bound on timestamp (inclusive).
    pub from_timestamp: Option<u64>,
    /// Upper bound on timestamp (inclusive).
    pub to_timestamp: Option<u64>,
    /// If set, only return records that disclose *at least* these field
    /// indices (subset match — records may disclose additional fields).
    pub required_fields: Option<Vec<u32>>,
    /// If set, only return records whose `target` equals this value.
    pub target_filter: Option<Vec<u8>>,
}

impl AuditQuery {
    /// True if `record` passes all active filters.
    pub fn matches(&self, record: &DisclosureRecord) -> bool {
        if let Some(from) = self.from_timestamp {
            if record.timestamp < from {
                return false;
            }
        }
        if let Some(to) = self.to_timestamp {
            if record.timestamp > to {
                return false;
            }
        }
        if let Some(ref required) = self.required_fields {
            for &idx in required {
                if !record.discloses(idx) {
                    return false;
                }
            }
        }
        if let Some(ref target) = self.target_filter {
            match &record.target {
                Some(t) => {
                    if t != target {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }
}

/// Append-only audit trail for BBS+ disclosure proof creations.
///
/// Indexed by `credential_id` (raw bytes) for O(log n) retrieval.
/// All entries for a given credential are stored in insertion order.
pub struct AuditTrail {
    /// credential_id (bytes) → chronological list of disclosure records.
    entries: BTreeMap<Vec<u8>, Vec<DisclosureRecord>>,
    /// Total number of records ever logged (monotonically increasing).
    total_logged: u64,
}

impl AuditTrail {
    /// Create an empty audit trail.
    pub fn new() -> Self {
        AuditTrail {
            entries: BTreeMap::new(),
            total_logged: 0,
        }
    }

    /// Append a disclosure record to the trail.
    ///
    /// This is the primary write path.  It must be called every time a
    /// selective-disclosure proof is created via `BbsPresentation::create_presentation`.
    ///
    /// # Arguments
    /// - `credential_id` — identifies the credential being presented.
    /// - `timestamp`     — caller-supplied UNIX timestamp.
    /// - `disclosed_fields` — message indices revealed in the proof.
    /// - `target`        — optional verifier identifier.
    pub fn log_disclosure(
        &mut self,
        credential_id: &[u8],
        timestamp: u64,
        disclosed_fields: &[u32],
        target: Option<&[u8]>,
    ) {
        let record = DisclosureRecord::new(credential_id, timestamp, disclosed_fields, target);
        self.entries
            .entry(credential_id.to_vec())
            .or_insert_with(Vec::new)
            .push(record);
        self.total_logged = self.total_logged.saturating_add(1);
    }

    /// Append a pre-constructed `DisclosureRecord`.
    ///
    /// Useful when the caller needs to attach field labels or other metadata
    /// before logging.
    pub fn log_record(&mut self, record: DisclosureRecord) {
        let key = record.credential_id.clone();
        self.entries
            .entry(key)
            .or_insert_with(Vec::new)
            .push(record);
        self.total_logged = self.total_logged.saturating_add(1);
    }

    /// Return all disclosure records for `credential_id` in insertion order.
    ///
    /// Returns an empty slice if no disclosures have been logged for this
    /// credential.
    pub fn get_disclosure_audit_trail(&self, credential_id: &[u8]) -> &[DisclosureRecord] {
        match self.entries.get(credential_id as &[u8]) {
            Some(records) => records.as_slice(),
            None => &[],
        }
    }

    /// Return all disclosure records for `credential_id` that satisfy the
    /// filter in `query`, in insertion order.
    ///
    /// An owned `Vec` is returned because filtering requires a new allocation.
    pub fn query(
        &self,
        credential_id: &[u8],
        query: &AuditQuery,
    ) -> Vec<&DisclosureRecord> {
        match self.entries.get(credential_id as &[u8]) {
            None => Vec::new(),
            Some(records) => records.iter().filter(|r| query.matches(r)).collect(),
        }
    }

    /// Return a flat iterator over all records across all credentials,
    /// ordered by credential key (lexicographic), then insertion order.
    pub fn all_records(&self) -> impl Iterator<Item = &DisclosureRecord> {
        self.entries.values().flat_map(|v| v.iter())
    }

    /// Total records ever appended (never decrements).
    pub fn total_logged(&self) -> u64 {
        self.total_logged
    }

    /// Number of distinct credentials that have at least one record.
    pub fn credential_count(&self) -> usize {
        self.entries.len()
    }

    /// True if no records have been appended.
    pub fn is_empty(&self) -> bool {
        self.total_logged == 0
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const CRED_A: &[u8] = b"credential-1";
    const CRED_B: &[u8] = b"credential-2";

    #[test]
    fn test_log_and_retrieve_single_record() {
        let mut trail = AuditTrail::new();
        trail.log_disclosure(CRED_A, 1_000_000, &[0, 2], Some(b"verifier-xyz"));

        let records = trail.get_disclosure_audit_trail(CRED_A);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].timestamp, 1_000_000);
        assert_eq!(records[0].disclosed_fields, vec![0, 2]);
        assert_eq!(records[0].target.as_deref(), Some(b"verifier-xyz" as &[u8]));
    }

    #[test]
    fn test_multiple_disclosures_for_same_credential() {
        let mut trail = AuditTrail::new();
        trail.log_disclosure(CRED_A, 1_000, &[1], Some(b"v1"));
        trail.log_disclosure(CRED_A, 2_000, &[1, 2], Some(b"v2"));
        trail.log_disclosure(CRED_A, 3_000, &[0, 1, 2], None);

        let records = trail.get_disclosure_audit_trail(CRED_A);
        assert_eq!(records.len(), 3);
        assert_eq!(trail.total_logged(), 3);
    }

    #[test]
    fn test_empty_trail_returns_empty_slice() {
        let trail = AuditTrail::new();
        assert!(trail.get_disclosure_audit_trail(b"no-such-cred").is_empty());
        assert!(trail.is_empty());
    }

    #[test]
    fn test_separate_credentials_do_not_interfere() {
        let mut trail = AuditTrail::new();
        trail.log_disclosure(CRED_A, 1_000, &[0], None);
        trail.log_disclosure(CRED_B, 2_000, &[1], None);

        assert_eq!(trail.get_disclosure_audit_trail(CRED_A).len(), 1);
        assert_eq!(trail.get_disclosure_audit_trail(CRED_B).len(), 1);
        assert_eq!(trail.credential_count(), 2);
    }

    #[test]
    fn test_query_by_timestamp_range() {
        let mut trail = AuditTrail::new();
        trail.log_disclosure(CRED_A, 100, &[0], None);
        trail.log_disclosure(CRED_A, 200, &[0], None);
        trail.log_disclosure(CRED_A, 300, &[0], None);
        trail.log_disclosure(CRED_A, 400, &[0], None);

        let q = AuditQuery {
            from_timestamp: Some(150),
            to_timestamp: Some(350),
            ..Default::default()
        };
        let results = trail.query(CRED_A, &q);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].timestamp, 200);
        assert_eq!(results[1].timestamp, 300);
    }

    #[test]
    fn test_query_by_required_fields() {
        let mut trail = AuditTrail::new();
        // Discloses field 0 only
        trail.log_disclosure(CRED_A, 100, &[0], None);
        // Discloses fields 0 and 2
        trail.log_disclosure(CRED_A, 200, &[0, 2], None);
        // Discloses field 2 only
        trail.log_disclosure(CRED_A, 300, &[2], None);

        let q = AuditQuery {
            required_fields: Some(alloc::vec![0, 2]),
            ..Default::default()
        };
        let results = trail.query(CRED_A, &q);
        // Only the record that discloses BOTH 0 and 2 matches.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, 200);
    }

    #[test]
    fn test_query_by_target() {
        let mut trail = AuditTrail::new();
        trail.log_disclosure(CRED_A, 100, &[0], Some(b"alice"));
        trail.log_disclosure(CRED_A, 200, &[0], Some(b"bob"));
        trail.log_disclosure(CRED_A, 300, &[0], None);

        let q = AuditQuery {
            target_filter: Some(b"alice".to_vec()),
            ..Default::default()
        };
        let results = trail.query(CRED_A, &q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, 100);
    }

    #[test]
    fn test_query_combined_filters() {
        let mut trail = AuditTrail::new();
        trail.log_disclosure(CRED_A, 100, &[0, 1], Some(b"verifier"));
        trail.log_disclosure(CRED_A, 200, &[0, 1], Some(b"verifier"));
        trail.log_disclosure(CRED_A, 300, &[0], Some(b"verifier"));

        let q = AuditQuery {
            from_timestamp: Some(150),
            required_fields: Some(alloc::vec![1]),
            target_filter: Some(b"verifier".to_vec()),
            ..Default::default()
        };
        let results = trail.query(CRED_A, &q);
        // Only record at 200: after ts=150, has field 1, target=verifier.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, 200);
    }

    #[test]
    fn test_record_labels_are_preserved() {
        let mut trail = AuditTrail::new();
        let record = DisclosureRecord::new(CRED_A, 500, &[2], Some(b"verifier"))
            .with_label(2, b"graduation_year");
        trail.log_record(record);

        let records = trail.get_disclosure_audit_trail(CRED_A);
        assert_eq!(
            records[0].field_labels.get(&2).map(|v| v.as_slice()),
            Some(b"graduation_year" as &[u8])
        );
    }

    #[test]
    fn test_disclosed_fields_are_sorted_and_deduplicated() {
        let record = DisclosureRecord::new(CRED_A, 0, &[3, 1, 2, 1, 3], None);
        assert_eq!(record.disclosed_fields, vec![1, 2, 3]);
    }

    #[test]
    fn test_discloses_helper() {
        let record = DisclosureRecord::new(CRED_A, 0, &[0, 2, 4], None);
        assert!(record.discloses(0));
        assert!(record.discloses(2));
        assert!(record.discloses(4));
        assert!(!record.discloses(1));
        assert!(!record.discloses(3));
    }

    #[test]
    fn test_all_records_iterator() {
        let mut trail = AuditTrail::new();
        trail.log_disclosure(CRED_A, 1, &[0], None);
        trail.log_disclosure(CRED_A, 2, &[1], None);
        trail.log_disclosure(CRED_B, 3, &[2], None);

        let count = trail.all_records().count();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_audit_trail_completeness() {
        // Simulate a realistic workflow: 5 disclosures for the same
        // credential, then verify the full trail matches exactly.
        let mut trail = AuditTrail::new();
        let events: &[(u64, &[u32], Option<&[u8]>)] = &[
            (1_000, &[0], Some(b"employer-a")),
            (2_000, &[0, 1], Some(b"employer-b")),
            (3_000, &[1], None),
            (4_000, &[0, 1, 2], Some(b"regulator")),
            (5_000, &[], None),
        ];

        for &(ts, fields, target) in events {
            trail.log_disclosure(CRED_A, ts, fields, target);
        }

        let records = trail.get_disclosure_audit_trail(CRED_A);
        assert_eq!(records.len(), events.len());
        assert_eq!(trail.total_logged(), events.len() as u64);

        for (i, &(ts, fields, target)) in events.iter().enumerate() {
            assert_eq!(records[i].timestamp, ts);
            let mut sorted_fields: Vec<u32> = fields.to_vec();
            sorted_fields.sort_unstable();
            assert_eq!(records[i].disclosed_fields, sorted_fields);
            assert_eq!(
                records[i].target.as_deref(),
                target.map(|t| t as &[u8])
            );
        }
    }
}
