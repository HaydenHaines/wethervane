"""Separate prelabel archived F0 and analytic PO5 action; no outcome reader."""

import execution_sources as sources
import numpy as np
from runner import utc, write_json, write_npz
from scipy.linalg import cho_factor, cho_solve

LEVELS = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)


def analytic_po5(likelihood, n):
    """Independent final-logit N(0,6.25), exact Gaussian likelihood update."""
    indices = np.asarray(likelihood.race_indices, int)
    h = np.eye(n)[indices]
    root_r = cho_factor(likelihood.covariance, lower=True)
    rh, ry = cho_solve(root_r, h), cho_solve(root_r, likelihood.values)
    precision = np.eye(n) / 6.25 + h.T @ rh
    rhs = h.T @ ry
    root = cho_factor(precision, lower=True)
    covariance = cho_solve(root, np.eye(n))
    mean = cho_solve(root, rhs)
    residuals = dict(
        covariance=float(np.linalg.norm(precision @ covariance - np.eye(n)) / np.linalg.norm(np.eye(n))),
        mean=float(np.linalg.norm(precision @ mean - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)),
        likelihood_matrix=float(np.linalg.norm(likelihood.covariance @ rh - h) / np.linalg.norm(h)),
        likelihood_values=float(
            np.linalg.norm(likelihood.covariance @ ry - likelihood.values)
            / max(np.linalg.norm(likelihood.values), np.finfo(float).tiny)
        ),
    )
    sources.require(
        max(residuals.values()) <= 1e-10
        and np.isfinite(mean).all()
        and np.isfinite(covariance).all()
        and (np.diag(covariance) > 0).all(),
        "selected analytic PO5 relative solve residual gate failed",
    )
    return (
        mean,
        covariance,
        dict(
            prior="independent final logit Normal(0,6.25)",
            solves="Cholesky; no sampled posterior",
            residuals=residuals,
            max_relative_solve_residual=1e-10,
        ),
    )


def interval_log_probability(mixture, lower, upper):
    loglower, _ = mixture.logtails(lower)
    logupper, logsfupper = mixture.logtails(upper)
    _, logsflower = mixture.logtails(lower)
    a, b = (logupper, loglower) if logupper <= logsflower else (logsflower, logsfupper)
    if a == b:
        return -np.inf
    return float(a + np.log(-np.expm1(b - a)))


def po5_reference(likelihood, race_ids, common_polled, modules):
    from numerical import component_means

    mean, covariance, solve = analytic_po5(likelihood, len(race_ids))
    sd = np.sqrt(np.diag(covariance))
    expected, integration = component_means(mean[None, None, :], sd[None, None, :], modules["math"])
    rows = []
    for j, name in enumerate(race_ids):
        mix = modules["math"].GaussianLogitMixture(mean[j : j + 1], sd[j : j + 1])
        quantiles = {str(q): mix.quantile(q)[0] for q in LEVELS}
        logwin, loglose = mix.log_win_lose()
        lognear = interval_log_probability(mix, 0.49, 0.51)
        loglarge = float(np.logaddexp(mix.logtails(0.45)[0], mix.logtails(0.55)[1]))
        near, large = float(np.exp(lognear)), float(np.exp(loglarge))
        rows.append(
            dict(
                race_id=name,
                common_polled=name in common_polled,
                expected_share=float(expected[0, 0, j]),
                expected_margin=float(2 * expected[0, 0, j] - 1),
                median_margin=2 * quantiles["0.5"] - 1,
                log_win=logwin,
                log_lose=loglose,
                quantiles=quantiles,
                p_absolute_margin_le_02=near,
                p_absolute_margin_ge_10=large,
                log_p_absolute_margin_le_02=lognear if np.isfinite(lognear) else None,
                log_p_absolute_margin_ge_10=loglarge,
            )
        )
    return dict(
        schema="MomentumAnalyticPO5PrelabelReferenceV1",
        rows=rows,
        solve=solve,
        quadrature=integration,
        numerical_integration_only_no_MCMC=True,
        core_or_Gibbs_ESS_not_applicable=True,
        common_polled_race_ids=list(common_polled),
        numeric_target_labels=False,
        scores=False,
    ), dict(mean_logit=mean, covariance=covariance, race_ids=np.asarray(race_ids))


def archived_f0_reference(shares, race_ids, posterior_draw, replicate, common_polled, modules):
    """Supplied archived values only; five future replicates remain row-clustered."""
    shares = np.asarray(shares)
    n = len(race_ids)
    sources.require(
        shares.shape == (4, 20000, n) and np.isfinite(shares).all() and ((shares >= 0) & (shares <= 1)).all(),
        "complete bounded archived F0 shares required",
    )
    sources.require(
        np.array_equal(posterior_draw, np.repeat(np.arange(4000), 5))
        and np.array_equal(replicate, np.tile(np.arange(5), 4000)),
        "source row/replicate identity required",
    )
    common_index = np.array([race_ids.index(name) for name in common_polled], int)
    pooled = shares.reshape(-1, n)
    ordered = np.sort(pooled, axis=0)
    quantiles = ordered[np.ceil(len(pooled) * np.asarray(LEVELS)).astype(int) - 1]
    wins = shares > 0.5
    clustered_win = wins.reshape(4, 4000, 5, n).mean(axis=2)
    clustered_mean = shares.reshape(4, 4000, 5, n).mean(axis=2)
    with np.errstate(divide="ignore"):
        log_win = np.log(clustered_win)
        log_lose = np.log1p(-clustered_win)
    names = list(common_polled)
    paired = modules["paired"].paired_tail_diagnostics(log_win[..., common_index], log_lose[..., common_index], names)
    win_diag, lose_diag = paired["win"], paired["lose"]
    mean_diag = modules["trace"].mean_share_diagnostics(clustered_mean[..., common_index], names)
    p = wins.mean(axis=(0, 1))
    source_p = wins.mean(axis=1)
    common = np.zeros(n, dtype=bool)
    common[common_index] = True
    competitive = common & (p >= 0.20) & (p <= 0.80)
    near_even = common & (p >= 0.40) & (p <= 0.60)
    rows = []
    for j, name in enumerate(race_ids):
        rows.append(
            dict(
                race_id=name,
                common_polled=bool(common[j]),
                win_probability=float(p[j]),
                per_source_chain_win_probability=source_p[:, j].tolist(),
                mask_boundary_distances={str(b): float(p[j] - b) for b in (0.20, 0.40, 0.60, 0.80)},
                expected_share=float(pooled[:, j].mean()),
                expected_margin=float(2 * pooled[:, j].mean() - 1),
                median_margin=float(2 * quantiles[3, j] - 1),
                quantiles={str(q): float(quantiles[k, j]) for k, q in enumerate(LEVELS)},
                p_absolute_margin_le_02=float(np.mean((pooled[:, j] >= 0.49) & (pooled[:, j] <= 0.51))),
                p_absolute_margin_ge_10=float(np.mean((pooled[:, j] <= 0.45) | (pooled[:, j] >= 0.55))),
            )
        )
    return dict(
        schema="MomentumArchivedF0PrelabelReferenceV1",
        rows=rows,
        common_polled_race_ids=names,
        competitive_race_ids=[r for j, r in enumerate(race_ids) if competitive[j]],
        near_even_race_ids=[r for j, r in enumerate(race_ids) if near_even[j]],
        mask_rule="inclusive .20/.80 and .40/.60 archived empirical D win; intersect fixed common-polled",
        win_side=win_diag,
        lose_side=lose_diag,
        tail_pairing=paired["pairing"],
        mean_share=mean_diag,
        uncertainty="five replicates clustered within each original posterior row; four source chains preserved",
        WIS_uncertainty="source-chain dispersion only; no certified nonlinear WIS MCSE",
        quantile_rule="sorted_x[ceil(N*q)-1]",
        numeric_target_labels=False,
        scores=False,
    ), dict(
        sorted_empirical_shares=ordered,
        joint_empirical_shares=shares,
        race_ids=np.asarray(race_ids),
        quantile_levels=np.asarray(LEVELS),
        quantiles=quantiles,
        row_clustered_win=clustered_win,
        row_clustered_share=clustered_mean,
        common_polled_mask=common,
        competitive_mask=competitive,
        near_even_mask=near_even,
    )


def actual_references(receipt, receipt_pin, modules):
    """Root-only later action. Author tests never call this on actual inputs."""
    from pathlib import Path

    from runner import validate_receipt

    bound, _, _, _ = validate_receipt(receipt_pin["path"], receipt_pin["sha256"])
    sources.require(bound == receipt and receipt["action"] == "references", "root reference receipt identity required")

    output = Path(receipt["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import os
    import traceback

    write_json(
        output / "launch.json",
        dict(
            schema="MomentumPrelabelReferenceLaunchV1",
            utc=utc(),
            pid=os.getpid(),
            ppid=os.getppid(),
            run_id=receipt["run_id"],
            execution_receipt=receipt_pin,
            synthetic=False,
        ),
    )
    artifacts = []
    try:
        for cycle in receipt["cycles"]:
            data = modules["adapter"].load_adapter(cycle)
            identity = modules["adapter"].archive_reference(cycle, data)
            _, fold, _ = modules["adapter"].TARGET.load_prepared(
                modules["adapter"].BINDINGS["files"][1]["path"],
                modules["adapter"].BINDINGS["files"][1]["sha256"],
                cycle,
                "native_full_series",
            )
            common = tuple(fold["common_polled_race_ids"])
            races = tuple(data.prior_kwargs["race_ids"])
            # This is the sole archived predictive numeric-value loading path.
            # It is separate from job execution and requires its own root receipt.
            sources.checked(identity["prediction_bytes"])
            with np.load(identity["prediction_bytes"]["path"], allow_pickle=False) as archive:
                sources.require(archive["race_ids"].tolist() == list(races), "archived reference roster drift")
                summary, arrays = archived_f0_reference(
                    archive["shares"], races, archive["posterior_draw"], archive["replicate"], common, modules
                )
            folder = output / str(cycle)
            folder.mkdir()
            summary["source_identity"] = data.source_identity
            summary["archived_identity"] = identity
            record = dict(cycle=cycle)
            artifacts.append(record)
            f0_arrays = write_npz(folder / "archived-F0-cdf.npz", arrays)
            record["archived_F0_arrays"] = f0_arrays
            write_json(folder / "archived-F0-summary.json", summary)
            record["archived_F0_summary"] = sources.pin(folder / "archived-F0-summary.json")
            likelihood, _ = data.build_likelihood("FS5")
            po5, po5arrays = po5_reference(likelihood, races, common, modules)
            po5["source_identity"] = data.source_identity
            po5["shared_FS5_input"] = data.measurements["FS5"].source_pins
            po5_arrays = write_npz(folder / "PO5-analytic.npz", po5arrays)
            write_json(folder / "PO5-summary.json", po5)
            record["PO5_arrays"] = po5_arrays
            record["PO5_summary"] = sources.pin(folder / "PO5-summary.json")
        terminal = dict(
            schema="MomentumPrelabelReferencesTerminalV1",
            status="complete_prelabel_references",
            utc=utc(),
            run_id=receipt["run_id"],
            execution_receipt=receipt_pin,
            artifacts=artifacts,
            masks_frozen_before_target_labels=True,
            numeric_target_labels=False,
            scores=False,
            no_new_F0_integration_or_draws=True,
        )
        write_json(output / "terminal.json", terminal)
        return terminal
    except BaseException as error:
        write_json(
            output / "terminal.json",
            dict(
                schema="MomentumPrelabelReferencesTerminalV1",
                status="failed_preserved_no_restart",
                utc=utc(),
                run_id=receipt["run_id"],
                execution_receipt=receipt_pin,
                artifacts=artifacts,
                exception_type=type(error).__name__,
                exception=str(error),
                traceback=traceback.format_exc(),
                numeric_target_labels=False,
                scores=False,
            ),
        )
        raise
