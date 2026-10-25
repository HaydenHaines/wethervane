"""Recompute raw-chain parameter diagnostics without sampling."""

import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "economic-execution-v1/attempt-v1"
records = []
for year in (2020, 2022, 2024):
    for arm in ("H0", "F0", "E01", "E01_sd025"):
        directory = RUN / str(year) / arm
        if not (directory / "terminal.json").exists():
            continue
        terminal = json.loads((directory / "terminal.json").read_bytes())
        banks = []
        for pin in terminal["raw_chains"]:
            assert hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest() == pin["sha256"]
            banks.append(az.from_netcdf(pin["path"]))
        assert len(banks) == 4
        for chain, bank in enumerate(banks):
            assert bank.posterior.sizes["draw"] == 4000
            assert bank.warmup_posterior.sizes["draw"] == 1500
            assert bank.posterior.chain.values.tolist() == [chain]
        joint = az.concat(*banks, dim="chain")
        names = ["beta", "tau_cycle", "tau_race", "cycle_z"]
        values = {}
        for metric, method in [("rhat", "rank"), ("bulk", "bulk"), ("tail", "tail")]:
            result = (
                az.rhat(joint, var_names=names, method=method)
                if metric == "rhat"
                else az.ess(joint, var_names=names, method=method)
            )
            raw = np.concatenate([np.asarray(result[name]).ravel() for name in names])
            assert np.isfinite(raw).all()
            values[metric] = float(raw.max() if metric == "rhat" else raw.min())
        values["divergences"] = int(joint.sample_stats.diverging.sum())
        values["depth_hits"] = int((joint.sample_stats.tree_depth >= 10).sum())
        assert values["rhat"] <= 1.01 and min(values["bulk"], values["tail"]) >= 400
        assert values["divergences"] == values["depth_hits"] == 0
        expected = json.loads((directory / "fit-diagnostics.json").read_bytes())["values"]
        for key, field in [
            ("rhat", "max_rank_rhat"),
            ("bulk", "min_bulk_ess"),
            ("tail", "min_tail_ess"),
            ("divergences", "divergences"),
            ("depth_hits", "max_depth_hits"),
        ]:
            np.testing.assert_allclose(values[key], expected[field], rtol=1e-12, atol=1e-12)
        records.append(dict(year=year, arm=arm, recomputed=values, banks=terminal["raw_chains"]))
(HERE / "diagnostics.json").write_text(json.dumps(dict(records=records, sampler_invoked=False), indent=2) + "\n")
print("Independent parameter diagnostics pass:", len(records), "jobs")
