"""Bank/measurement and byte/metadata-only predictive-reference checks; no transitions."""

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import adapter as a


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = {year: a.load_adapter(year) for year in (2018, 2020, 2022, 2024)}
        cls.plan = json.loads(a.sources.checked(a.BINDINGS["files"][8]))
        cls.bank = next(b for b in a.COMPATIBILITY["banks"] if b["year"] == 2018 and b["arm"] == "F0")
        cls.design = json.loads(a.sources.checked(cls.adapters[2018].source_pins["target_fold_design"]))

    def test_all_original_joint_rows_equal_raw_source(self):
        for year, data in self.adapters.items():
            bank = next(b for b in a.COMPATIBILITY["banks"] if b["year"] == year and b["arm"] == "F0")
            for chain, source in enumerate(bank["raw_banks"]):
                handle = a.az.from_netcdf(source["path"])
                try:
                    start = chain * 4000
                    for name in a.POSTERIOR_VARIABLES:
                        a.np.testing.assert_array_equal(
                            data.original_parameter_rows[name][start : start + 4000], handle.posterior[name].values[0]
                        )
                finally:
                    handle.close()
            a.np.testing.assert_array_equal(data.prior_kwargs["posterior_chain"], a.np.repeat(a.np.arange(4), 4000))
            a.np.testing.assert_array_equal(data.prior_kwargs["posterior_draw"], a.np.tile(a.np.arange(4000), 4))
            self.assertEqual(data.prior_kwargs["beta"].shape, (16000, 6))
            a.np.testing.assert_array_equal(data.uniform_prior_weights, a.np.full(16000, 1 / 16000))
            self.assertNotIn("cycle_z", data.prior_kwargs)
            self.assertEqual(data.prior_kwargs["race_df"], 5.0)

    def test_immutable_parameters_coordinates_and_constructed_cache(self):
        data = self.adapters[2018]
        prior, _ = data.build_prior()
        likelihood, _ = data.build_likelihood("FS5")
        for array in (
            data.prior_kwargs["beta"],
            data.uniform_prior_weights,
            prior.beta,
            prior._beta_products,
            prior.posterior_draw,
            likelihood.covariance,
            data.measurements["FS5"].coordinates.days_to_election,
        ):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(TypeError):
            data.prior_kwargs["source_identity"] = "other"
        with self.assertRaises(TypeError):
            data.source_pins["raw_banks"][0]["sha256"] = "other"
        a.np.testing.assert_array_equal(prior.coefficients, prior.beta)
        self.assertEqual(prior.source_identity, data.source_identity)

    def test_shared_full_series_latest_subset_covariance_and_spd(self):
        for year, data in self.adapters.items():
            fs5, fs6, lp = [data.measurements[name] for name in ("FS5", "FS6", "LP5")]
            self.assertEqual(fs5.coordinates.observation_ids, fs6.coordinates.observation_ids)
            a.np.testing.assert_array_equal(fs5.shares, fs6.shares)
            full = fs5.coordinates.observation_ids
            subset = [full.index(obs) for obs in lp.coordinates.observation_ids]
            a.np.testing.assert_allclose(
                lp.likelihood_kwargs["covariance"],
                fs5.likelihood_kwargs["covariance"][a.np.ix_(subset, subset)],
                rtol=0,
                atol=0,
            )
            self.assertEqual(dict(fs5.scales), dict(lp.scales))
            self.assertEqual(fs5.scales["persistent_linear"], 0)
            self.assertGreater(fs6.scales["persistent_linear"], 0)
            for measurement in (fs5, fs6, lp):
                a.np.linalg.cholesky(measurement.likelihood_kwargs["covariance"])
                self.assertEqual(len(measurement.shares), len(measurement.coordinates.observation_ids))
                self.assertEqual(
                    measurement.source_pins["calibration"]["calibration_target"], 2022 if year == 2024 else year
                )
                self.assertTrue(measurement.source_pins["no_nominal_variance_added"])
        self.assertNotEqual(self.adapters[2022].source_identity, self.adapters[2024].source_identity)

    def test_bad_design_law_coefficients_and_future_training_rejected(self):
        for key, change in [
            ("law_hash", "wrong"),
            ("coefficient_names", list(reversed(self.design["coefficient_names"]))),
            ("race_ids", list(reversed(self.design["race_ids"]))),
            ("offset", [v + 0.01 for v in self.design["offset"]]),
        ]:
            design = copy.deepcopy(self.design)
            design[key] = change
            with self.assertRaises(ValueError):
                a.read_joint_bank(self.bank, design, self.plan)
        bank = copy.deepcopy(self.bank)
        bank["training_cycles"].append(2018)
        with self.assertRaisesRegex(ValueError, "future training"):
            a.read_joint_bank(bank, self.design, self.plan)

    def test_raw_axis_variable_draw_and_scale_corruption_rejected(self):
        original = a.az.from_netcdf

        def broken(kind):
            def load(path):
                handle = original(path)
                if kind == "coefficient":
                    handle.posterior = handle.posterior.isel(coefficient=slice(None, None, -1))
                if kind == "draw":
                    handle.posterior = handle.posterior.isel(draw=slice(1, None))
                if kind == "chain":
                    handle.posterior = handle.posterior.assign_coords(chain=[99])
                if kind == "missing":
                    handle.posterior = handle.posterior.drop_vars("tau_race")
                if kind == "scale":
                    handle.posterior = handle.posterior.load()
                    handle.posterior["tau_race"].values[0, 0] = -1.0
                return handle

            return load

        for kind in ("coefficient", "draw", "chain", "missing", "scale"):
            with self.subTest(kind=kind), mock.patch.object(a.az, "from_netcdf", broken(kind)):
                with self.assertRaises(ValueError):
                    a.read_joint_bank(self.bank, self.design, self.plan)

    def test_chain_receipt_seed_and_identity_are_required(self):
        original = a.trusted_json

        def wrong(path):
            value = original(path)
            if str(path).endswith("chain-0-complete.json"):
                value["seed"] += 1
            return value

        with mock.patch.object(a, "trusted_json", wrong), self.assertRaisesRegex(ValueError, "receipt/seed/job"):
            a.read_joint_bank(self.bank, self.design, self.plan)

    def test_wrong_constructor_pin_and_module_rejected(self):
        data = self.adapters[2018]
        wrong = dict(a.trusted_pin(a.OWNER / "src/forecasting/polling_basis_v1.py"))
        wrong["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not reviewed"):
            data.build_prior(constructor_pin=wrong)
        with self.assertRaisesRegex(ValueError, "constructor name"):
            data.build_prior(prior_class=a.native_basis.GaussianPollLikelihood)

    def test_private_sampler_constructors_use_exact_snapshot_classes(self):
        root = a.LAB / "momentum-observed-sampler-sol-v1"
        response_path = root / "snapshots/src/forecasting/response_v5_reuse.py"
        polling_path = root / "snapshots/src/forecasting/polling_v1.py"
        observed_path = root / "observed_sampler.py"
        old = sys.modules.get("src.forecasting.polling_v1")
        response = a.sources.load_bound_module("adapter_test_response", a.trusted_pin(response_path))
        deps = {"src.forecasting.response_v5_reuse": response}
        polling = a.sources.load_bound_module("adapter_test_polling", a.trusted_pin(polling_path), deps)
        deps["src.forecasting.polling_v1"] = polling
        observed = a.sources.load_bound_module("adapter_test_observed", a.trusted_pin(observed_path), deps)

        def no_sampling(frame, event, arg):
            if event == "call" and frame.f_code is observed.sample_poll_posterior.__code__:
                raise AssertionError("Sampler forbidden in adapter test")

        sys.setprofile(no_sampling)
        try:
            prior, likelihood, binding = self.adapters[2018].build_for_sampler(
                observed,
                "FS5",
                a.trusted_pin(observed_path),
                [a.trusted_pin(response_path), a.trusted_pin(polling_path)],
            )
        finally:
            sys.setprofile(None)
        self.assertIs(sys.modules.get("src.forecasting.polling_v1"), old)
        self.assertIsInstance(prior, observed.EmpiricalBasisFundamentals)
        self.assertNotIsInstance(prior, a.native_basis.EmpiricalBasisFundamentals)
        self.assertIsInstance(likelihood, observed.GaussianPollLikelihood)
        self.assertIs(
            observed.sample_poll_posterior.__globals__["GaussianPollLikelihood"], polling.GaussianPollLikelihood
        )
        self.assertTrue(binding["sampler_not_called"])
        with self.assertRaisesRegex(ValueError, "constructor source pins"):
            self.adapters[2018].build_for_sampler(observed, "FS5", a.trusted_pin(observed_path), [])

    def test_archive_reader_reads_only_identity_vectors(self):
        original = a.np.lib.format.read_array
        read_names = []

        def guarded(stream, *args, **kwargs):
            self.assertIn(stream.name, ("race_ids.npy", "posterior_draw.npy", "replicate.npy"))
            read_names.append(stream.name)
            return original(stream, *args, **kwargs)

        with mock.patch.object(a.np.lib.format, "read_array", guarded):
            for year, data in self.adapters.items():
                record = a.archive_reference(year, data)
                self.assertFalse(record["probability_or_logit_values_read"])
                self.assertEqual(record["predictive_realizations"], 80000)
                self.assertEqual(record["parameter_rows"], 16000)
        self.assertEqual(len(read_names), 12)

    def test_archive_alignment_shape_dtype_and_order_corruption(self):
        data = self.adapters[2018]
        seal = a.trusted_json(a.LAB / "economic-execution-v1/attempt-v1/2018/F0/prediction-seal.json")
        headers, metadata = a.read_archive_metadata(seal["predictions"]["path"])
        for key in ("race_ids", "posterior_draw", "replicate"):
            altered = {name: value.copy() for name, value in metadata.items()}
            if key == "race_ids":
                altered[key] = altered[key][::-1]
            else:
                altered[key][0] += 1
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                a.validate_archive_metadata(headers, altered, data.prior_kwargs["race_ids"])
        for change in ("shape", "dtype", "fortran_order"):
            bad = copy.deepcopy(headers)
            bad["shares"][change] = [4, 20000, 1] if change == "shape" else "int64" if change == "dtype" else True
            with self.assertRaises(ValueError):
                a.validate_archive_metadata(bad, metadata, data.prior_kwargs["race_ids"])
        # Synthetic malformed NPZ: no actual predictive numeric array is loaded.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.npz"
            a.np.savez(path, race_ids=a.np.array(["synthetic"]))
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                a.read_archive_metadata(path)

    def test_source_tampering_rejected_before_module_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy(a.HERE / "source-bindings.json", tmp / "source-bindings.json")
            shutil.copytree(a.HERE / "snapshots", tmp / "snapshots")
            file = next((tmp / "snapshots").iterdir())
            file.write_bytes(file.read_bytes() + b"changed")
            with mock.patch.object(a.sources, "load_bound_module", side_effect=AssertionError("premature import")):
                with self.assertRaisesRegex(ValueError, "source drift"):
                    a.sources.verify(tmp)
        self.assertEqual(
            a.BINDINGS["files"][12]["sha256"], "45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678"
        )

    def test_undeclared_receipt_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as tmp:
            with self.assertRaisesRegex(ValueError, "undeclared adapter"):
                a.trusted_json(Path(tmp) / "self-rebound.json")
            with self.assertRaises(FileExistsError):
                a.report(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
