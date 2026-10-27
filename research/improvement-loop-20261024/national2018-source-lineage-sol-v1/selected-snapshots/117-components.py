"""Real prior/expression/factor owners for the two exact baseline laws."""

import hashlib
import inspect
from datetime import date
from pathlib import Path

import numpy as np
from scipy.linalg import helmert

from src.senate_count_stack_v1.demographics import CELL_IDS
from src.senate_count_stack_v1.measurement import masked_tensor
from .emission import emission_tensor, shared_epsilon
from .intended import first_preference_tensor, reported_pair_tensor

# Loaded as a research package by api.py; no modifications of frozen modules.
from . import graph, references


def stage(name, inputs=(), outputs=()):
    return graph.Stage(name, tuple(graph.PORTS[k] for k in inputs), tuple(graph.PORTS[k] for k in outputs))


def pooled(name, shape, scale):
    import pymc as pm

    sd = pm.HalfNormal(name + "_sd", sigma=scale)
    raw = pm.Normal(name + "_raw", 0, 1, shape=shape)
    return raw * sd


class Provider:
    role = ""
    implementation_id = ""
    stages = ()
    parameter_patterns = ()

    def __init__(self):
        source = Path(inspect.getfile(type(self))).resolve()
        self.spec = graph.ComponentSpec(
            self.implementation_id,
            self.role,
            self.stages,
            self.parameter_patterns,
            str(source),
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )

    def build(self, stage_name, context, scope):
        return getattr(self, stage_name)(context, scope)


class ProtectedPopulation(Provider):
    role, implementation_id = "population", "protected_full_parent_exposure_v1"
    stages = (stage("exposure", outputs=("exposure",)),)

    def exposure(self, c, scope):
        return dict(exposure=np.array(c.protected["exposure"]))


class Community(Provider):
    role, implementation_id = "community", "demographic_profile_drift_v1"
    stages = (stage("composition", outputs=("community_logits", "community_temporal", "community")),)
    parameter_patterns = (r"community_(intercept|main|interaction_(sd|raw)|state_(sd|raw)|drift_raw)",)

    def composition(self, c, scope):
        import pymc as pm
        import pytensor.tensor as pt

        intercept = pm.Normal("community_intercept", 0, 1.5, shape=11)
        main = pm.Normal("community_main", 0, 0.5, shape=(6, 11))
        interaction = pooled("community_interaction", (5, 11), 0.25)
        state = pooled("community_state", (len(c.states), 11), 0.5)
        base = (
            intercept[None, None, :]
            + pt.dot(c.x[:, :6], main)[None, :, :]
            + pt.dot(c.x[:, 6:], interaction)[None, :, :]
            + state[:, None, :]
        )
        if len(c.dates) > 1:
            elapsed = np.array(
                [
                    (date.fromisoformat(b) - date.fromisoformat(a)).days / 365.25
                    for a, b in zip(c.dates[:-1], c.dates[1:], strict=True)
                ]
            )
            raw = pm.Normal("community_drift_raw", 0, 1, shape=(len(c.dates) - 1, 11))
            drift = pt.concatenate((pt.zeros((1, 11)), pt.cumsum(raw * (0.05 * np.sqrt(elapsed))[:, None], axis=0)))
        else:
            drift = pt.zeros((1, 11))
        logits, temporal = pt.dot(base, c.z.T), pt.dot(drift, c.z.T)
        values = pm.Deterministic(
            "community",
            pm.math.softmax(logits[c.state_axis] + temporal[c.event_dates, None, :], axis=-1),
            dims=("event", "demographic", "community_profile"),
        )
        return dict(community_logits=logits, community_temporal=temporal, community=values)


class Environment(Provider):
    role, implementation_id = "environment", "separated_lkj2_cycle_student5_race_v1"
    stages = (stage("residuals", outputs=("cycle_effect", "race_effect")),)
    parameter_patterns = (r"cycle_(sd|correlation_unit|raw)", r"final_race_(sd|raw)")

    def residuals(self, c, scope):
        import pymc as pm
        import pytensor.tensor as pt

        chol = references.cycle.build_cycle_factor()
        raw = pm.Normal("cycle_raw", 0, 1, shape=(len(c.cycles), 2))
        effect = pm.Deterministic("cycle_effect", pt.dot(raw, chol.T))
        if c.contests:
            sd = pm.HalfNormal("final_race_sd", 0.25)
            race = pm.StudentT("final_race_raw", nu=5, mu=0, sigma=1, shape=len(c.contests)) * sd
        else:
            race = pt.zeros((0,))
        scope.physical_diagnostics.extend(references.cycle.PHYSICAL_DIAGNOSTICS)
        scope.annotations.update(
            cycle_coordinate_version=references.cycle.VERSION, physical_cycle_cholesky_is_deterministic=True
        )
        return dict(cycle_effect=effect, race_effect=race)


class Fundamentals(Provider):
    role, implementation_id = "fundamentals", "lean_carryover_context_incumbency_v1"
    stages = (stage("mean", outputs=("structural_choice", "incumbency")),)
    parameter_patterns = (
        r"choice_(intercept|relative_lean|national_carryover|president_midterm)",
        r"candidate_incumbency",
    )

    def mean(self, c, scope):
        import pymc as pm

        if not c.contests:
            return dict(structural_choice={}, incumbency=0.0)
        intercept = pm.Normal("choice_intercept", 0, 0.25)
        lean = pm.Normal("choice_relative_lean", 1, 0.35)
        national = pm.Normal("choice_national_carryover", 1, 0.25)
        midterm = pm.Normal("choice_president_midterm", 0, 0.25)
        incumbent = pm.Normal("candidate_incumbency", 0.30, 0.25)
        values = {
            k["id"]: intercept
            + lean * k["relative_lean"]
            + national * k["national_pres_logit"]
            + midterm * c.context[c.event_index[k["event_id"]]]
            for k in c.contests
        }
        return dict(structural_choice=values, incumbency=incumbent)


class Participation(Provider):
    role, implementation_id = "participation", "profile_turnout_response_observed_total_gamma_v1"
    stages = (
        stage(
            "behavior",
            ("exposure", "community", "cycle_effect"),
            ("turnout_logit", "propensity", "cell_intensity", "event_gamma_shape", "response_logits", "response"),
        ),
        stage("event", ("cell_intensity", "event_gamma_shape", "base_category_rates"), ("eventG", "event_intensity")),
    )
    parameter_patterns = (
        r"turnout_(intercept|midterm|main|(interaction|community|state)_(sd|raw))",
        r"response_(intercept|race_(sd|raw))",
        r"event_gamma_shape",
        r"event_observed_total_coordinate",
        r"eventG_unobserved",
    )

    def behavior(self, c, scope):
        import pymc as pm
        import pytensor.tensor as pt

        v = scope.inputs
        intercept = pm.Normal("turnout_intercept", np.log(0.65 / 0.35), 0.6)
        midterm = pm.Normal("turnout_midterm", -0.4, 0.4)
        main = pm.Normal("turnout_main", 0, 0.25, shape=6)
        interaction = pooled("turnout_interaction", 5, 0.15)
        community = pooled("turnout_community", 11, 0.20)
        state = pooled("turnout_state", len(c.states), 0.30)
        demo = pt.dot(c.x[:, :6], main) + pt.dot(c.x[:, 6:], interaction)
        logit = (
            intercept
            + midterm * c.midterm[:, None, None]
            + demo[None, :, None]
            + pt.dot(c.z, community)[None, None, :]
            + state[c.state_axis, None, None]
            + v["cycle_effect"][c.cycle_axis, 0, None, None]
        )
        q = pm.Deterministic("propensity", pm.math.sigmoid(logit), dims=("event", "demographic", "community_profile"))
        intensity = v["exposure"][:, :, None] * v["community"] * q
        kappa = pm.LogNormal("event_gamma_shape", np.log(100), 0.7)
        logits, response = {}, {}
        if c.contests:
            baseline = pm.Normal("response_intercept", np.log(0.98 / 0.02), 0.5)
            race = pooled("response_race", len(c.contests), 0.3)
            for i, contest in enumerate(c.contests):
                name = contest["id"]
                logits[name] = baseline + race[i]
                response[name] = pm.Deterministic(f"response_{name}", pm.math.sigmoid(logits[name]))
        return dict(
            turnout_logit=logit,
            propensity=q,
            cell_intensity=intensity,
            event_gamma_shape=kappa,
            response_logits=logits,
            response=response,
        )

    def event(self, c, scope):
        import pymc as pm
        import pytensor.tensor as pt

        v = scope.inputs
        official = {c.contest_map[o["contest_id"]]["event_id"]: o for o in c.data["official_counts"]}
        observed = [e["id"] for e in c.events if e["id"] in official]
        unobserved = [e["id"] for e in c.events if e["id"] not in official]
        logs, totals, gs = [], [], {}
        for event in observed:
            o = official[event]
            contest = c.contest_map[o["contest_id"]]
            ids = [x["id"] for x in contest["candidates"] + contest["noncandidate_categories"]]
            indices = [ids.index(k) for k in o["counts"]]
            logs.append(pt.log(pt.sum(v["base_category_rates"][o["contest_id"]][indices])))
            totals.append(sum(o["counts"].values()))
        centers = np.log(np.array(totals, dtype=float) + 0.5) if c.composition.standardized else np.zeros(len(totals))
        scales = (
            1 / np.sqrt(np.array(totals, dtype=float) + 0.5) if c.composition.standardized else np.ones(len(totals))
        )
        if observed:
            log_base = pm.Deterministic("observed_log_base_total", pt.stack(logs))
            coordinate = pm.Flat(
                "event_observed_total_coordinate", shape=len(observed), initval=np.zeros(len(observed))
            )
            log_total = pm.Deterministic("observed_log_expected_total", centers + scales * coordinate)
            log_g = log_total - log_base
            g = pt.exp(log_g)
            name = "event_observed_gamma_density_and_jacobian"
            scope.factor(name, kind="coordinate_prior")
            pm.Potential(
                name,
                pt.sum(
                    pm.logp(pm.Gamma.dist(alpha=v["event_gamma_shape"], beta=v["event_gamma_shape"]), g)
                    + log_g
                    + np.log(scales)
                ),
            )
            gs.update({name: g[i] for i, name in enumerate(observed)})
        if unobserved:
            g = pm.Gamma(
                "eventG_unobserved", alpha=v["event_gamma_shape"], beta=v["event_gamma_shape"], shape=len(unobserved)
            )
            gs.update({name: g[i] for i, name in enumerate(unobserved)})
        g = pm.Deterministic("eventG", pt.stack([gs[e["id"]] for e in c.events]), dims="event")
        intensity = pm.Deterministic("event_intensity", g * v["cell_intensity"].sum(axis=(1, 2)), dims="event")
        scope.physical_diagnostics.append("eventG")
        scope.annotations.update(
            coordinate_version="observed_total_scaled_v1" if c.composition.standardized else "observed_log_total_v1",
            observed_event_indices=tuple(c.event_index[e] for e in observed),
            unobserved_event_indices=tuple(c.event_index[e] for e in unobserved),
            coordinate_centers=tuple(centers.tolist()),
            coordinate_scales=tuple(scales.tolist()),
            physical_eventG_is_deterministic=True,
        )
        return dict(eventG=g, event_intensity=intensity)


def within_bloc_basis(candidates, bloc):
    names = sorted(c["id"] for c in candidates if c["bloc"] == bloc)
    result = np.zeros((len(candidates), max(len(names) - 1, 0)))
    if len(names) > 1:
        basis = helmert(len(names)).T
        for row, candidate in enumerate(candidates):
            if candidate["id"] in names:
                result[row] = basis[names.index(candidate["id"])]
    return result


class Choice(Provider):
    role, implementation_id = "choice", "candidate_iid_profile_choice_v1"
    stages = (
        stage(
            "utility",
            ("structural_choice", "incumbency", "cycle_effect", "race_effect"),
            ("candidate_utilities", "candidate_probability", "political_axes"),
        ),
    )
    parameter_patterns = (
        r"choice_(main|(interaction|community|state)_(sd|raw))",
        r"candidate_contrast_.+",
        r"candidate_other_.+",
        r"candidate_role_(mean|sd)",
    )
    role_hierarchy = False

    def utility(self, c, scope):
        import pymc as pm
        import pytensor.tensor as pt

        v = scope.inputs
        utilities, probabilities, axes = {}, {}, {}
        role_rows, observed, physical, noncentered = {}, set(), [], []
        if self.role_hierarchy:
            roles = references.roles.validate_effective_binding(c.problem, c.roles)
            role_rows = {(r["contest_id"], r["candidate_id"]): r for r in roles.to_dict()["candidates"]}
            observed = {(o["contest_id"], k) for o in c.data["official_counts"] for k in o["counts"]}
            pooled_keys = {
                (t["id"], k["id"])
                for t in c.contests
                for k in t["candidates"]
                if k["bloc"] == "other"
                and not (len(t["candidates"]) == 2 and sum(x["bloc"] == "other" for x in t["candidates"]) == 1)
            }
            if pooled_keys:
                role_mean = pm.Normal("candidate_role_mean", -3, 3, shape=2)
                role_sd = pm.HalfNormal("candidate_role_sd", 1, shape=2)
        elif c.roles is not None:
            raise ValueError("IID candidate provider cannot silently consume a role binding")
        if c.contests:
            main = pm.Normal("choice_main", 0, 0.5, shape=6)
            interaction = pooled("choice_interaction", 5, 0.25)
            community = pooled("choice_community", 11, 0.30)
            state = pooled("choice_state", len(c.states), 0.20)
            demo = pt.dot(c.x[:, :6], main) + pt.dot(c.x[:, 6:], interaction)
            for ci, contest in enumerate(c.contests):
                name, candidates = contest["id"], contest["candidates"]
                ei = c.event_index[contest["event_id"]]
                political = np.array([0.5 if k["bloc"] == "D" else -0.5 if k["bloc"] == "R" else 0 for k in candidates])
                axes[name] = political
                eta = (
                    v["structural_choice"][name]
                    + demo[:, None]
                    + pt.dot(c.z, community)[None, :]
                    + state[c.state_axis[ei]]
                    + v["cycle_effect"][c.cycle_axis[ei], 1]
                    + v["race_effect"][ci]
                )
                utility = eta[:, :, None] * political[None, None, :]
                adjustment = v["incumbency"] * np.array([k["incumbent"] for k in candidates], dtype=float)
                for bloc in ("D", "R"):
                    basis = within_bloc_basis(candidates, bloc)
                    if basis.shape[1]:
                        deviation = pm.Normal(f"candidate_contrast_{name}_{bloc}", 0, 0.5, shape=basis.shape[1])
                        adjustment = adjustment + pt.dot(basis, deviation)
                others = [i for i, k in enumerate(candidates) if k["bloc"] == "other"]
                sole = len(candidates) == 2 and len(others) == 1
                for index in others:
                    person = f"candidate_other_{name}_{candidates[index]['id']}"
                    if not self.role_hierarchy:
                        minor = pm.Normal(person, -0.3 if sole else -3, 1)
                    else:
                        physical.append(person)
                        key = (name, candidates[index]["id"])
                        if sole:
                            minor = pm.Normal(person, -0.3, 1)
                        else:
                            role = references.roles.ROLES.index(role_rows[key]["ballot_role"])
                            if c.composition.person_coordinates == "centered_v1" or key in observed:
                                minor = pm.Normal(person, role_mean[role], role_sd[role])
                            else:
                                raw = pm.Normal(person + "_raw", 0, 1)
                                minor = pm.Deterministic(person, role_mean[role] + role_sd[role] * raw)
                                noncentered.append(person)
                    adjustment = adjustment + minor * np.eye(len(candidates))[index]
                utilities[name] = utility + adjustment[None, None, :]
                probabilities[name] = pm.Deterministic(
                    f"candidate_probability_{name}", pm.math.softmax(utilities[name], axis=-1)
                )
        scope.physical_diagnostics.extend(physical)
        if self.role_hierarchy:
            scope.annotations.update(
                role_binding_identity=roles.identity,
                role_manifest_sha256=roles.manifest_sha256,
                person_coordinate_version=c.composition.person_coordinates,
                physical_person_intercepts=tuple(physical),
                noncentered_person_intercepts=tuple(noncentered),
                role_rebind_proof_sha256=None
                if roles.rebind_proof_bytes is None
                else references.roles.digest(roles.rebind_proof_bytes),
                role_admission_cutoff_utc=roles.admission_cutoff_utc,
                role_axis=references.roles.ROLES,
            )
        return dict(candidate_utilities=utilities, candidate_probability=probabilities, political_axes=axes)


class RoleChoice(Choice):
    implementation_id = "candidate_TX_ballot_role_profile_choice_v1"
    role_hierarchy = True


class CountsRules(Provider):
    role, implementation_id = "counts_rules", "single_mark_category_allocation_poisson_v1"
    stages = (
        stage(
            "allocation",
            ("cell_intensity", "candidate_probability", "response"),
            ("category_probability", "base_category_rates"),
        ),
        stage("counts", ("eventG", "base_category_rates"), ("category_rates",)),
    )
    parameter_patterns = (r"noncandidate_allocation_.+",)

    def allocation(self, c, scope):
        import pymc as pm
        import pytensor.tensor as pt

        v = scope.inputs
        categories, rates = {}, {}
        for contest in c.contests:
            name = contest["id"]
            width = len(contest["noncandidate_categories"])
            # Substantive option-allocation prior, explicitly owned by this provider.
            allocation = pm.Dirichlet(f"noncandidate_allocation_{name}", np.ones(width)) if width > 1 else pt.ones((1,))
            allocation = pm.Deterministic(f"noncandidate_probability_{name}", allocation)
            response = v["response"][name]
            category = pt.concatenate(
                (
                    response * v["candidate_probability"][name],
                    pt.broadcast_to((1 - response) * allocation, (24, 12, width)),
                ),
                axis=-1,
            )
            categories[name] = pm.Deterministic(f"category_probability_{name}", category)
            rates[name] = pt.sum(
                v["cell_intensity"][c.event_index[contest["event_id"]], :, :, None] * categories[name], axis=(0, 1)
            )
        return dict(category_probability=categories, base_category_rates=rates)

    def counts(self, c, scope):
        import pymc as pm

        v = scope.inputs
        rates = {}
        for contest in c.contests:
            name = contest["id"]
            rates[name] = pm.Deterministic(
                f"category_intensity_{name}",
                v["eventG"][c.event_index[contest["event_id"]]] * v["base_category_rates"][name],
            )
        for o in c.data["official_counts"]:
            contest = c.contest_map[o["contest_id"]]
            names = [x["id"] for x in contest["candidates"] + contest["noncandidate_categories"]]
            indices = [names.index(k) for k in o["counts"]]
            name = f"official_likelihood_{o['id']}"
            scope.factor(name, [f"official:{o['id']}"])
            pm.Poisson(
                name, mu=rates[o["contest_id"]][indices], observed=np.array(list(o["counts"].values()), dtype=np.int64)
            )
        return dict(category_rates=rates)


class Measurement(Provider):
    role, implementation_id = "measurement", "joint_CES_ranked_intended_report_F1a_pair_v1"
    stages = (
        stage(
            "observations",
            (
                "community_logits",
                "community_temporal",
                "turnout_logit",
                "candidate_utilities",
                "political_axes",
                "response_logits",
                "event_intensity",
                "category_rates",
                "cell_intensity",
            ),
        ),
    )
    parameter_patterns = (
        r"source_(community|community_state|turnout|choice|response|not_sure|false_vote)",
        r"report_(release|state)",
        r"source_answer_error_.+",
    )

    def observations(self, c, scope):
        import pymc as pm
        import pytensor.tensor as pt

        v = scope.inputs
        channels = c.problem.to_dict()["report_channels"]
        epsilon_cache = {}
        ranked_config = c.problem.to_dict()["ranked_measurement"]
        ranked_styles = {r["contest_id"]: r["style"] for r in ranked_config["ranked"]}
        if c.source_names:
            qn = len(c.source_names)
            community = pm.Normal("source_community", 0, 0.25, shape=(qn, 11))
            state = pm.Normal("source_community_state", 0, 0.15, shape=(qn, len(c.states), 11))
            turnout = pm.Normal("source_turnout", 0.3, 0.3, shape=qn)
            choice = pm.Normal("source_choice", 0, 0.2, shape=qn)
            response_shift = pm.Normal("source_response", 0, 0.5, shape=qn)
            unsure = pm.Beta("source_not_sure", 1, 19, shape=qn)
            false_vote = pm.Beta("source_false_vote", 1, 49, shape=qn)
            for survey in c.data["surveys"]:
                ei = c.event_index[survey["event_id"]]
                si, qi = c.state_axis[ei], c.source_axis[survey["source_id"]]
                logits = (
                    v["community_logits"][si]
                    + v["community_temporal"][c.date_index[survey["community_date"]]][None, :]
                    + pt.dot(c.z, community[qi] + state[qi, si])[None, :]
                )
                w = pm.math.softmax(logits, axis=-1)
                h = pm.math.sigmoid(v["turnout_logit"][ei] + turnout[qi])
                rows = survey["rows"]
                cells = np.array([CELL_IDS.index(r["cell_id"]) for r in rows])
                cm, hm = np.array([r["community_mask"] for r in rows]), np.array([r["turnout_mask"] for r in rows])
                if survey["contest_id"] is None:
                    table = w[:, :, None] * pt.stack((1 - h, h), axis=-1)
                    probability = pt.sum(table[cells] * cm[:, :, None] * hm[:, None, :], axis=(1, 2))
                else:
                    name = survey["contest_id"]
                    individual = pm.math.softmax(
                        v["candidate_utilities"][name] + choice[qi] * v["political_axes"][name][None, None, :], axis=-1
                    )
                    if name in ranked_styles:
                        individual = first_preference_tensor(individual, ranked_styles[name], array_module=pt)
                    matrix = np.array(
                        [
                            [k["id"] in g["candidate_ids"] for g in survey["groups"]]
                            for k in c.contest_map[name]["candidates"]
                        ],
                        dtype=float,
                    )
                    grouped = pt.dot(individual, matrix)
                    response = pm.math.sigmoid(v["response_logits"][name] + response_shift[qi])
                    channel = channels[survey["id"]]
                    epsilon = shared_epsilon(channel, epsilon_cache, pm_module=pm)
                    table = emission_tensor(w, h, individual, response, unsure[qi], false_vote[qi], channel, epsilon, array_module=pt)
                    probability = masked_tensor(
                        table, cells, cm, hm, np.array([r["answer_mask"] for r in rows]), array_module=pt
                    )
                name = f"survey_likelihood_{survey['id']}"
                scope.factor(name, [f"survey:{survey['source_id']}:{r['record_id']}" for r in rows])
                pm.Potential(name, pt.sum(pt.log(probability)))
        reports = c.data["participation_reports"]
        if reports:
            releases = sorted({r["release_id"] for r in reports})
            release = pm.Normal("report_release", 0, 0.03, shape=len(releases))
            state = pm.Normal("report_state", 0, 0.05, shape=len(c.states))
            for r in reports:
                ei = c.event_index[r["event_id"]]
                bias = release[releases.index(r["release_id"])] + state[c.state_axis[ei]] - (0.03**2 + 0.05**2) / 2
                name = f"report_likelihood_{r['id']}"
                scope.factor(name, [f"report:{r['id']}"])
                pm.NegativeBinomial(name, mu=v["event_intensity"][ei] * pt.exp(bias), alpha=100, observed=r["count"])
        for block in c.data["polls"]:
            means = []
            for row in block["rows"]:
                names = [k["id"] for k in c.contest_map[row["contest_id"]]["candidates"]]
                first, second = [names.index(row[k]) for k in ("first_candidate_id", "second_candidate_id")]
                if row['observation_id'] in ranked_config['poll_channels']:
                    pc = ranked_config['poll_channels'][row['observation_id']]
                    cid = row['contest_id']; ei = c.event_index[c.contest_map[cid]['event_id']]
                    worth = pm.math.softmax(v['candidate_utilities'][cid] + pc['source_choice_shift'] * v['political_axes'][cid][None,None,:], axis=-1)
                    epsilon = shared_epsilon(pc['channel'], epsilon_cache, pm_module=pm)
                    _, logit, _ = reported_pair_tensor(worth, ranked_styles[cid], pc['channel'], epsilon,
                        v['cell_intensity'][ei], pm.math.sigmoid(v['response_logits'][cid]), np.array(pc['selection_weights']),
                        pc['first_label'], pc['second_label'], array_module=pt)
                    means.append(logit)
                else:
                    rates = v["category_rates"][row["contest_id"]]
                    means.append(pt.log(rates[first]) - pt.log(rates[second]))
            name = f"poll_likelihood_{block['id']}"
            scope.factor(name, [f"poll:{r['observation_id']}" for r in block["rows"]])
            pm.MvNormal(
                name,
                mu=pt.stack(means),
                cov=np.array(block["covariance"]),
                observed=np.array([r["logit_value"] for r in block["rows"]]),
            )
        return {}


def standard_registry():
    result = graph.Registry()
    for provider in (
        ProtectedPopulation(),
        Community(),
        Environment(),
        Fundamentals(),
        Participation(),
        Choice(),
        RoleChoice(),
        CountsRules(),
        Measurement(),
    ):
        result.register(provider)
    return result


def baseline_composition(*, role_hierarchy=False, standardized=True, person_coordinates="hybrid_v1"):
    providers = (
        ProtectedPopulation,
        Community,
        Fundamentals,
        Environment,
        Participation,
        RoleChoice if role_hierarchy else Choice,
        CountsRules,
        Measurement,
    )
    return graph.Composition(tuple((p.role, p.implementation_id) for p in providers), standardized, person_coordinates)
