# Independent protected-parent cut challenge

Material correction required to the proposed next test, not to the completed two-state numerical comparison. No graph, fit, source/attempt mutation or statistical library import. All inspected source hashes are in source-pins.json; check.py/result.json preserve a bounded exact-rational counterexample.

Owner `research/handoffs-20261009/cut-contract-v1.json:177–185` is explicit:

- K_i(psi)=pi_D(psi|U_i)L_joint(...|U_i,psi).
- Z_i=integral K_i(psi)dpsi, with 0<Z_i<infinity.
- q_i(psi)=K_i(psi)/Z_i.
- p_cut(U_i,psi)=w_i q_i(psi), with original upstream mass w_i.
- Prediction uses original upstream weights and normalized conditional weights (line183).
- Global normalization that changes upstream weights proportional to Z_i is expressly forbidden (line185).

The implementation agrees: `src/senate_count_stack_v1/joint.py:35` says never reweight U; modular `graph.py:538–539` declares population_cut_v1 and parent_mass_used_in_logp=False. `banks.py:140` accepts no evidence/Z argument; lines164–170 assign conditional_row_weight=1/(chains×draws), joint_row_weight=parent.mass/(chains×draws), and marginal_likelihood_reweighting=False. These are source semantics, not a newly inferred approval rule.

Luna review `m01-compressed-actual-equivalence-luna-review-v1/review.json:46–48` instead proposes conditional log-evidence calculation, including parent mass once, then normalizing parent weights with log-sum-exp. If those weights mean exp(log(w_i)+log(Z_i)−logsumexp_j(log(w_j)+log(Z_j))), they become w_i Z_i/sum_j w_j Z_j. This is the forbidden globally normalized joint, not population_cut_v1. The fact each original mass is included once does not fix the target change. The existing review's supported fixed-state observations remain valid; its 'integrating/updating across all eight parents' wording at line37 should also clarify that protected parent masses are never updated by downstream evidence under this cut.

Synthetic counterexample: eight parents have original masses1/8. Seven kernels over two states are (1,1), the eighth is (2,2), so Z=(2,...,2,4). All eight conditional laws q_i are exactly(1/2,1/2). The cut keeps all parent masses1/8. Evidence normalization changes seven masses to1/9 and the eighth to2/9. A future observable equal to1 only in the final parent then has probability1/8 under the cut and2/9 under the proposed weighting, despite identical within-parent conditional laws. All kernels are finite and positive, so neither divergence nor a zero-mass exception explains the difference. This is algebraic illustration, not actual model evidence.

A corrected next test may compare the two representations across all eight fixed parents, conditional posterior calculations and fixed-mass predictive mixtures. If importance sampling is selected, normalize proposal weights *within each parent*: a_ij=(K_i(psi_ij)/g_i(psi_ij))/sum_j(K_i(psi_ij)/g_i(psi_ij)); final paired weight is w_i a_ij. Validate support, finite normalizers, ESS/error and original whole-state pairing independently. Estimated Z_i can be a diagnostic/comparison of kernel representation, but must not update w_i. Kernel equality at finitely many states or a noisy evidence estimate cannot establish global equivalence or adequate conditional sampling. Full all-parent inference is a separate costly test, not automatically the highest-value immediate step merely because two-state parity passed.

An intentionally uncut evidence-weighted alternative would require a separately versioned scientific target and review; it cannot be described as maintaining the protected cut. Preserve the original review and issue a correction/addendum. No actual all-parent computation or predictive qualification occurred here.
