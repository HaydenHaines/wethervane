# ADR-005: Shift-Based Community Discovery

## Status
Accepted (2026-03-18)

## Context

The original two-stage approach to community discovery relied on a separation between non-political and political data:

1. **Stage 1**: Discover latent community types from non-political features using soft assignment (NMF on ACS demographics + RCMS religious data)
2. **Stage 2**: Validate that these types predict political covariance by fitting historical election results

This approach achieved R²=0.66 on historical elections (2016, 2018, 2020), confirming the core hypothesis: communities that share social identity and behavioral patterns do covary politically.

However, the two-stage approach has limitations:

- **Indirection**: NMF communities are defined by demographic similarity, which may not capture the most politically relevant community boundaries. Communities that shift together politically may share non-obvious demographic characteristics.
- **Prescriptive clustering**: The model imposes a demographic structure and then validates it politically. This prevents the model from discovering the community structure that is *most* predictive of political behavior.
- **Limited expressiveness**: NMF produces soft clusters defined by weighted combinations of demographic features. This structure may not match the actual political organization of space.

## Decision

Invert the discovery process: define communities directly from spatially correlated electoral shift patterns.

**New approach:**

1. **Shift vectors**: For each census tract, compute a 9-dimensional electoral shift vector capturing changes in Democratic %, Republican %, and turnout across three election pairs:
   - 2016→2020 presidential
   - 2020→2024 presidential
   - 2018→2022 midterm

2. **Spatial adjacency**: Build a Queen-contiguity graph of census tracts (tracts sharing an edge or corner are neighbors)

3. **Hierarchical clustering**: Apply hierarchical agglomerative clustering (linkage: average or complete) on shift vectors, with the spatial contiguity constraint: two clusters can merge only if they are geographically adjacent

4. **Community discovery**: Clusters converge to communities that:
   - Shift together politically (low within-cluster variance in shift vectors)
   - Are geographically contiguous (satisfy Queen adjacency)

5. **Demographic overlay**: After discovery, overlay demographics (ACS features, RCMS religious composition, LODES commuting patterns, IRS migration flows) descriptively to characterize the discovered communities

## Consequences

### Positive

- **Direct discovery**: Communities are defined by how they move politically, not demographic proxies. This makes the discovery process more mechanistic and easier to validate.
- **Falsifiability through temporal holdout**: Train on pre-2024 shifts (2016→2020, 2020→2024), test on 2024 actual shifts. This provides a clean temporal validation path: communities that predict historical shifts well must also predict out-of-sample 2024 shifts.
- **Geographic coherence**: Spatial contiguity constraint ensures discovered communities are geographically contiguous, which is essential for practical forecasting and interpretation.
- **Inverted assumptions**: Demographic questions now flow *from* the discovered communities rather than dictating community structure. E.g., "What is the demographic profile of the communities that shift together?" rather than "Do demographic groups covary politically?"

### Negative / Trade-offs

- **Two-stage separation no longer applies**: Communities are now defined by political behavior, not non-political features. This removes the ability to claim "non-political features predict covariance" — instead, we claim "communities defined by electoral shift patterns can be characterized by non-political features and predict future shifts."
- **Increased data dependency**: The discovery is now directly based on election returns (the outcome data), not upstream non-political data. This requires careful temporal validation to avoid overfitting.
- **Potential loss of insight**: The original NMF approach produced interpretable community types (e.g., "Asian suburban," "Black urban," "evangelical rural"). The shift-based approach produces geographically defined communities that may be harder to name and describe.
- **Smaller datasets**: Only three election pairs yield 9-dimensional shift vectors. This is not a large sample for unsupervised clustering. Regularization or dimensionality reduction may be needed.

### Relationship to historical approach

The original two-stage NMF approach remains valuable for comparison and validation:

- Retain the NMF code path (shelved in `src/detection/`) for comparison
- After community discovery via shifts, check whether the discovered communities align with NMF types
- Use NMF validation results (R²=0.66) as a baseline — shift-based communities should achieve at least this accuracy on historical data

### Implementation plan

1. Implement shift vector computation (`src/discovery/shift_vectors.py`)
2. Build spatial adjacency matrix (`src/discovery/adjacency.py`)
3. Implement HAC with spatial constraints (`src/discovery/cluster.py`)
4. Overlay demographics (`src/description/overlay_demographics.py`)
5. Temporal validation: 2024 holdout test
6. Compare to NMF baseline

See the shift-community-discovery plan at `docs/superpowers/plans/2026-03-18-shift-community-discovery.md` for detailed task breakdown.
