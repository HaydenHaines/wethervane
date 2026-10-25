"""SYNTHETIC ONLY durable deliberate failure; never loads actual target inputs."""

from pathlib import Path
from types import SimpleNamespace

import runner
from test_execution import MODULES, fixture, provenance, small_job


def main():
    prior, likelihood = fixture()
    original = MODULES["sampler"].sample_poll_posterior

    def fail(prior, polls, *, observer, **kwargs):
        def callback(record):
            observer(record)
            if record["phase"] == "completed_sweep":
                raise RuntimeError("SYNTHETIC deliberate observer-prefix failure before original output assignment")

        return original(prior, polls, observer=callback, **kwargs)

    modules = {**MODULES, "sampler": SimpleNamespace(sample_poll_posterior=fail)}
    output = Path(__file__).resolve().parent / "synthetic-failure-v1"
    try:
        runner.run_prepared_job(
            prior, likelihood, modules, small_job(10), output, provenance=provenance(), synthetic=True
        )
    except RuntimeError as error:
        if "SYNTHETIC deliberate" not in str(error):
            raise
        print(runner.sources.pin(output / "terminal.json"))
    else:
        raise AssertionError("Expected synthetic failure was not raised")


if __name__ == "__main__":
    main()
