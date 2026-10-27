# Correction: selected response shape in candidate allocation oracle review

This separately versioned addendum corrects one material statement in the preserved V1 review. The V1 claim that the selected provider's response was cell-specific was wrong. In pinned 117-components.py, lines 193–200, response is one scalar per contest: sigmoid(response_intercept + response_race[i]). CountsRules lines 415–425 broadcasts it over cell-specific candidate probabilities/intensities.

Therefore the oracle's one fixed response shared across its synthetic profiles matches the frozen source shape. Its profile weights can represent source cell-intensity weights (including the common event scale), with the correct candidate probabilities and noncandidate simplex. The earlier heterogeneous-response counterexample is still mathematically correct for a different hypothetical source, but it is not a limitation or counterexample to this pinned provider.

I retract the source-shape criticism only. The examples remain invented expected allocations; no model state, raw rows or outcomes were used. They do not identify a causal layer, establish historical calibration, or equate candidate marks with physical turnout. No owner/model work was changed. Source pins bind this correction to both the preserved V1 bytes and the exact provider source.
