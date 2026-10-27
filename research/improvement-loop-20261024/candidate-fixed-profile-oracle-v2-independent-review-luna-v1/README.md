# Independent V2 review: fixed-profile allocation oracle

V2's expected-allocation arithmetic matches the declared formula. All 11 bounded standard-library tests pass. I independently checked the two-profile heterogeneous-response example against exact fractions: named expectations are 124/3, 16/3 and 160/3; noncandidate expectations are 25 and 75; total intensity 200 is conserved; aggregate D/R is 31/4. The common scalar-response special case also agrees with the pinned provider shape.

The source correction is resolved: selected 117-components.py uses one scalar response per contest, broadcast over cell-specific choice probabilities and intensity. V2 says its per-profile response is hypothetical, so it does not overclaim source parity. The original V1 review and its separately pinned correction remain unchanged.

These invented results establish an allocation identity only. They do not use model state, outcomes or AZ rows; they do not identify a causal source for the Arizona score discrepancy or establish turnout, calibration or forecast performance. Use synthetic intensity/participant weights and expected category marks as terminology.

Exact subject, model-source, V1 correction, test and probe hashes are in source-pins.json.
