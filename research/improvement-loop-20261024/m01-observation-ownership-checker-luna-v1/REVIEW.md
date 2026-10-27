# Independent review: observation-ownership-checker-sol-v1

## Verdict

The successor fixes the main schema defects from my earlier critique. It now separates event/target context from original observation identity, distinguishes different fields on one record, rejects the same declared atom in separately multiplied factors, restricts within-factor atom reuse to an explicitly joint structure, separates typed shared inputs from observed atoms, and keeps sampling/source-error independence unauthenticated. The A2018 adapter is labeled aggregate consistency only. I found one concrete chronology gap in the claim checker: it accepts an observation period ending after the declared source availability time.

This is a consistency issue in caller-supplied metadata, not evidence that any actual source is false. All results continue to disclaim authentication and scientific qualification.

## Frozen inputs and reproducibility

The reviewed packet is observation-ownership-checker-sol-v1. Pins supplied and checked against its evidence record:

- checker.py SHA256 468c745bf76056a48e9641a4570141f753749067853a45124155def2957b8532
- test_checker.py SHA256 2c06f773166f1531d022aee3e30593865df9c0c0551f32a9ca9689ed07e5fe1a
- evidence.json SHA256 96bedd30d324a664aabcb9ed0a6bbb90a079055e459e06a1ea9e8323a59c5a83

I copied checker.py and test_checker.py to a fresh temporary directory and ran:

    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest test_checker -v

All 18 invented tests passed. The retained output is tests.log. No model, posterior or prediction arrays were opened; no owner job ran; no respondent token is present in this review packet.

## Findings

### Improvements verified by source and fixtures

- An observation atom is keyed by origin namespace, source-scoped record token, and raw field ID. Measurement ID, raw unit, event, and period are checked for consistency. Target and model-unit transformations do not alter the original atom key, preventing a target relabel or transformed unit from hiding repeated evidence.
- Two fields from the same record are treated as distinct observations. The test leaves their sampling overlap unknown rather than calling them independent.
- The same declared original atom in two separate likelihood factors sets duplicate_declared_observed_atoms. Repeating it within one component rejects. Reuse across components is accepted only when the caller declares a joint factor and supplies a shaped joint-model pin; outputs expressly leave joint-law authentication false.
- Deterministic inputs and uncertain inputs are separate source roles. Shared covariates are not counted as observed evidence; reused uncertain-input references are flagged. Observed/derived-observed inputs to priors require review.
- A coherent fabricated lineage and sample attestation can still pass structural checks, but the result remains unauthenticated and unqualified. This is intentional and clearly documented.
- The A2018 checker pins the frozen aggregate and validates summary counts only. It reports record_ownership_audited=false and sampling_overlap=unknown. The underlying A2018 summary has 65 survey blocks/7,968 memberships and one 274-row covariance poll block; for Arizona it has 20 poll rows and no direct survey/count/report. Those facts do not identify unique survey respondents or poll samples.

### Finding: impossible availability/period ordering is accepted

checker.audit verifies source.available_at <= manifest.cutoff and observation.period_end <= cutoff independently. It does not require source.available_at >= observation.period_end. The saved probe creates a fully resealed fixture with source available_at 2020-01-01, observation period ending 2020-01-15, and cutoff 2020-02-01. audit returns structurally_consistent_claims, authenticated fields false. Reproduce with:

    PYTHONDONTWRITEBYTECODE=1 python3 -B temporal-order-probe.py

If available_at is intended to mean when this source artifact containing the observed record became available, require period_end <= available_at (or encode an explicit exception/source role for data not released until later). Historical availability still requires external evidence; this guard would only reject an internally impossible declared timeline.

## Remaining limits, not blockers for the stated claim-only scope

- Hashes and provenance are caller-authored. The manifest digest protects against accidental mutation relative to a retained digest, but a caller can coherently rewrite both. Atom identity and source aliases are not derived by the checker.
- Sampling labels are supplied declarations, not linked to observation records or source bytes. Declared disjointness is counted but never verified; shared source-error dependence remains unknown for every cross-factor pair. The output correctly leaves verified_independence false.
- A joint-model hash does not prove normalization, correct covariance, or that the model graph uses that law. One factor with structure=joint is only a representation claim.
- The generic checker does not bind claims to actual selected factor code, full input source bytes, A2018's ordered poll vector/covariance, candidate roster, or parent bank. The A2018 aggregate check is specifically not record-level evidence authentication.
- Temporal checks only validate supplied timestamps and cutoff ordering. They do not authenticate public release/vintage. The A2018 source-level audit retains unknown poll vintage and respondent/sample linkage.

No actual A2018 observation atoms or respondent identities were reconstructed in this review. The tool is a stronger claim-consistency gate than the previous version, but a source-reading adapter remains necessary before factual ownership claims.

