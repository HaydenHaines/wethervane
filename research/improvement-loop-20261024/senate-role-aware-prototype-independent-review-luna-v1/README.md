# Independent review: Senate role-aware allocation prototype

I ran the seven role-aware tests from a temporary copy. All pass. An independent invented probe confirms the stated fixed-mass property: expanding from one to 32 eligible write-ins preserves aggregate write-in, major-pair, printed-other, and noncandidate mass; a 1,000-unit expected category allocation is conserved. Zero-support and observed-count partition behavior also match the documented rules.

The prototype uses explicit caller-supplied roles, eligibility, response, total nonmajor mass, write-in fraction, utilities, and noncandidate probabilities. It does not infer or validate any of those against official ballot records, estimate group-mass parameters, or fit the model. The selected model's iid other-utility softmax differs structurally; this prototype is an alternative law and cannot establish that it would improve Arizona or any other forecast.

No Arizona outcomes/raw predictions or owner work were used. Exact pins, test log, independent probe, and scope are in this packet.
