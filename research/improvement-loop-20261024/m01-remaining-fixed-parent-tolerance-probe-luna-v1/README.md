# V2 tolerance-validation challenge

`probe.py` builds the V2 publisher's invented dynamic-witness protocol fixture in a fresh temporary copy, changes one comparison error to `1e100`, then coherently updates the comparison pin, controller-success marker and retained caller response. `selection.completed()` accepts it. The probe does not import or run a model graph.

This confirms that `validate_comparisons()` checks exact comparison field/schema/status and finite nonnegative error values but does not independently enforce the comparison tolerances. The fixture is coherently self-authored: it has modified the publisher output and regenerated the local hashes and caller return. Under the documented cooperating-client model, this is same-user evidence fabrication, not an accidental hash mismatch. In the real production path, the reviewed comparison gate's numeric `close()` checks reject above-tolerance values before `publish_success`; its artifacts are then pinned, and ordinary accidental mutation is caught by pins and retained response equality. Therefore the probe does not establish a practical defect in the frozen cooperative workflow, though repeating tolerance or a comparator-pass assertion in `validate_comparisons()` would add defense in depth and make the helper's standalone contract stronger.

V2 remains unmodified. No graph or owner work ran.
