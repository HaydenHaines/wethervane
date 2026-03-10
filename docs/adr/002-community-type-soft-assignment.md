# ADR-002: Community Types as Soft Assignments (Mixture Model)

## Status
Accepted

## Context
A central modeling choice is how to assign counties to community types. Two approaches are possible:

**Hard assignment**: Each county belongs to exactly one community type. This is the output of clustering algorithms like Leiden, SBM, or k-means. A county labeled "evangelical rural" is treated as 100% that type.

**Soft assignment**: Each county is a mixture of community types, represented as a weight vector summing to 1. For example, a county might be [0.40 evangelical rural, 0.30 military retiree, 0.30 suburban professional]. This is the output of NMF, topic models (LDA), or mixed-membership models.

The choice has downstream consequences for every stage of the model:

- **Covariance estimation**: With hard assignment, each county contributes to exactly one type's election history. With soft assignment, each county's election results are decomposed across types.
- **Poll decomposition**: With hard assignment, a state poll's geographic footprint maps to a discrete set of types. With soft assignment, the poll is decomposed via spectral unmixing -- the poll result is modeled as a weighted sum of type-level opinions, where the weights come from the county mixture proportions within the poll's footprint.
- **Prediction aggregation**: With hard assignment, county prediction = type prediction. With soft assignment, county prediction = weighted sum of type predictions.

Counties are empirically heterogeneous. Duval County, FL (Jacksonville) contains urban Black communities, suburban white-collar neighborhoods, a major Navy base, and rural-adjacent areas. Forcing it into one type discards information. Conversely, small rural counties may be well-described by a single type, making soft assignment unnecessary overhead.

## Decision
Use soft assignment (mixture model / NMF framing) as the primary representation of county-to-type relationships.

Specifically:
- Community detection produces a county x type weight matrix W, where W[i,k] is the proportion of county i's "community character" attributable to type k, and each row sums to 1.
- The primary method for producing W is graph-regularized NMF on the county feature matrix, with the community network (commuting, migration, SCI) providing the graph regularization term.
- Hard assignments from Leiden/SBM clustering are retained as a secondary output for interpretability, comparison (Assumption A005), and as initialization for NMF.
- The number of types K is selected via cross-validated reconstruction error and political covariance quality (see Assumption A006).

## Consequences
**What becomes easier:**
- Heterogeneous counties (urban, suburban, mixed) are represented more accurately.
- Poll decomposition via spectral unmixing is mathematically natural: if a poll covers counties with known mixture weights, the poll result is a linear combination of type-level opinions (Assumption A008).
- The model can represent gradual transitions between community types (e.g., exurban counties that are partially rural, partially suburban) rather than forcing sharp boundaries.
- County-level predictions benefit from information across multiple types rather than depending entirely on one type's estimate.

**What becomes more difficult:**
- Covariance estimation must account for the mixing: type-level election results are not directly observed but must be inferred from mixed county-level results. This is an additional inference step.
- The spectral unmixing for polls requires solving a (potentially ill-conditioned) linear system for each poll.
- Interpretability is harder -- stakeholders may prefer a simple label ("this county is type X") over a mixture vector.
- Model complexity increases: the W matrix has N_counties x K_types free parameters (minus sum-to-one constraints), compared to N_counties categorical labels for hard assignment.
- Validating the soft assignments is harder than validating hard clusters, since there is no simple "correct" partition to compare against.
