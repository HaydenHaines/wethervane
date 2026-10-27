# Independent participation rational-simplex candidate V2 review

Frozen owner candidate implementation SHA98263eee20241bfff0516b32442ab730f8b283ef24edfb327b09ce6e673ef716 matches the README. Source and owner tests were copied to fresh temporary scratch; all13 owner tests pass0.002s. Five independent adversarial tests pass0.005s, including24 permutations of the rare-anchor case. Owner files and original V1 source pins remained byte-identical after review. No training/outcome data, banks, graph, sampler, forecast or owner job was used.

## Findings

No new defect reproduced in the bounded synthetic cases. Diff against the original owner prototype changes only mixture_moments/docstring: same near-unit admission gate, then explicit rational normalization and exact rational accumulation of conditional float means/variances before returning floats. Conditional PMF, cell moments, support validation and structural rank code are unchanged. This fixes the original tolerance-induced mixture support violation and avoids the first correction's low-weight-anchor cancellation. It is a deliberately different declared working-simplex policy, not permission to normalize external protected parent masses.

Independent tests:

* Prior rare anchor: N=2^53, tiny first weight9.555591182889165e−19 and three near-unit remainder weights. All24 scenario permutations exactly equal the float of the independent normalized Fraction mean and variance, with nonnegative supported mean. Owner test also covers N1million and reversed order.
* Minimum positive subnormal weight, represented as math.ulp(0), with a deterministic high-count scenario and unit-weight zero scenario: result equals the exact rational normalized reference converted to float. This shows rational mixing preserves a representable tiny contribution that an unstable centered subtraction could lose; it does not restore probabilities already rounded tozero upstream.
* Near-unit sums on both sides of1 at N0,1,1million,2^53, identical deterministic p0/p1 scenarios: exact endpoint means and variance0. Original observed excess/spurious variance is absent.
* Two invented events have opposite deterministic scenario outcomes. Exact Fraction means, positive variances and negative cross-covariance match; covariance is symmetric and splitting one scenario into two exactly representable equal weights leaves the full output unchanged.
* A p.5/p1 binomial mixture matches independent Fraction within+between variance. Nonfinite/negative/zero-total/outside-tolerance weights reject; a zero-weight finite scenario leaves the result unchanged.

The split test preserves the exact rational total represented by input floats. Arbitrarily splitting a float through separately rounded arithmetic need not preserve its binary value; the candidate correctly represents supplied float inputs, not intended decimal weights. Owner README explicitly acknowledges that limitation.

## Remaining boundaries

Fraction conversion happens after conditional_moments already computes sigmoid and weighted cell moments in float. It cannot recover finite-logit probabilities that saturate0/1, fractional input intentions, or lost conditional precision. “Exact” rational mixture accumulation is exact for these supplied float moments/weights only. Final float conversion rounds covariance/means; exact PSD of the rational covariance does not imply every downstream floating eigensolver reports a nonnegative minimum eigenvalue. No full numeric error bound, exhaustive domain proof, or real-data impact is established.

Conditional event independence given shared scenarios remains an assumption; this is shared-effect uncertainty, not longitudinal identities, population uncertainty or extra event-shock integration. Eligible populations are caller-supplied fixed integers; fractional uncertain projections cannot be rounded into voters without a separate scientific law. Observed flags/rank do not fit any count, identify an unseen cycle effect, authenticate data, or supply priors. Mutable caller collections remain an external stability assumption.

Work is O(S×E²) rational covariance terms plus conditional moment work; rational bit-cost and allocations exceed floating arithmetic. The7ms combined unit-suite timings are tiny fixture observations, not a scale benchmark or justification for a fit/backend. No large scenario/event cohort, actual model compilation or memory-throughput study was run. This remains a potentially useful synthetic correctness reference, unadmitted for scientific/production use.

Reproduce with a fresh copy of candidate participation_extrapolation_prototype.py plus its two owner test files: nice -n15 python3 -B -m unittest test_participation_extrapolation_prototype test_normalized_simplex -v. Add test_review.py from this review and run nice -n15 python3 -B -m unittest test_review -v. Full logs/source pins/diff retained. Preserve original prototype and failed first correction evidence; this review supersedes only the tested rational-simplex candidate's normalization/rare-anchor behavior, not prior scientific limitations. No forecast qualification/default adoption or outer-weight change follows.
