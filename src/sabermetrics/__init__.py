"""Political sabermetrics: advanced analytics for politician performance.

Separate silo within the US Political Covariation Model project.
Shares data infrastructure with the community-covariance pipeline
but has its own compute pipeline, outputs, and use cases.

Core concept: decompose election outcomes into district baseline +
national environment + candidate effect. The candidate effect IS the
stat. When the community-covariance model is available, the candidate
effect is further decomposed into a Community-Type Overperformance
Vector (CTOV) -- a per-community-type scouting report.

Submodules:
    ingest       -- Data download and ID crosswalk construction
    baselines    -- District baseline computation (Cook PVI, structural)
    residuals    -- Candidate residual computation (MVD, CTOV, polling gap)
    electoral    -- Electoral performance stats
    campaign     -- Campaign finance stats (SDR, FER, burn rate)
    legislative  -- Legislative effectiveness stats (LES, ASR, CSN, SDL)
    constituent  -- Constituent relationship stats (CAR, IAR, RI from CES)
    speech       -- Floor speech influence (NLP on congressional record)
    composites   -- Career summaries, fit scores, talent pipeline
"""
