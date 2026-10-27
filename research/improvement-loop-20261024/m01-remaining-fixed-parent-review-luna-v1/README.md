# Independent review: remaining fixed-parent verifier

The frozen six-parent verifier's own file pins all match. Its 40 bounded tiny tests pass in a fresh temporary copy. No actual graph, model, compiler, fit, sampler or owner job ran.

The staged design addresses the old FD mismatch correctly: for each selected parent 2–7, the dynamic-witness stage compares three saved dynamic calls and captures new same-parent finite differences; a separate fixed-native stage compares that witness to an independently compiled fixed-parent graph. Parent identity includes complete exposure; fixed masses remain 0.125. Comparisons are pointwise, not normalized posterior evidence.

I found a pre-launch acknowledgment gap. `selection.completed()` accepts a raw caller-supplied SHA of a success-shaped marker without requiring an actual retained `write_stdin_return` object, and it accepts a marker with an empty comparison-artifact map. The fixed-selected launcher uses that helper before launching the costly graph. I reproduced this with a tiny isolated fixture; later comparison gates should prevent a false completed comparison receipt, but run authorization has already been consumed. The README’s external acknowledgment requirement remains out-of-band at this boundary.

The required successor checks and exact reproduction are in `review.json` and `reproduce.py`. This is a workflow protocol gap within the cooperative local environment, not a claim of cryptographic same-user security. The original frozen verifier was not modified, and no actual-phase admission was created.
