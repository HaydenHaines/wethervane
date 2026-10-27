# Independent review: direct-executor ownership V1

I reran the 12 bounded tests (3.208 seconds) and independently compared top-level function ASTs with the pinned V3 executor. The source changes only `prepare`, `load`, and `registered`; launch, locks, inspect, supervision, child, terminal/recovery, stop, and collection functions remain unchanged. The fixture suite covers original launch/retry/live-worker behavior after the new ownership binding.

The global ownership lock serializes cooperating `prepare()` clients in one root. Component-aware pairwise checks reject exact, ancestor and descendant collisions in either order while permitting lexical siblings. Durable intent precedes scaffold publication, and crashes at each tested boundary preserve exact-intent repair without launching. Mixed source/store identities fail closed.

The explicit negative case remains material: if the only pre-scaffold intent is externally deleted, no attempt/output/job artifact remains to witness the claim, so a nested claim is accepted. This is not a normal process-crash transition under the stated local fsync assumption, but it means the registry has no independent protection against sole-intent loss. Add redundant/append-only evidence if that failure mode must be covered.

The work is scoped to cooperating successor clients for one root. Legacy clients, cross-root sharing, hostile writers, child filesystem sandboxing, power loss and scale are not covered. Full evidence and hashes are in `review.json`.
