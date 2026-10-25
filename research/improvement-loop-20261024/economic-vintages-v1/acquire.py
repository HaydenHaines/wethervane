"""Fetch unchanged BLS releases; preserve failed direct requests and FRASER copies.

No current-series substitution. Successful existing bytes are never overwritten.
Run explicitly to acquire/retry; parser/selector/tests are offline.
"""

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Dates resolved from BLS archive-index links, not inferred observation dates.
RELEASES = [
    ("1998-09", "1998-10-16", "history/cpi_101698.txt"),
    ("2000-09", "2000-10-18", "history/cpi_10182000.txt"),
    ("2002-09", "2002-10-18", "history/cpi_10182002.txt"),
    ("2004-09", "2004-10-19", "history/cpi_10192004.txt"),
    ("2006-09", "2006-10-18", "history/cpi_10182006.txt"),
    ("2008-09", "2008-10-16", "archives/cpi_10162008.htm"),
    ("2010-09", "2010-10-15", "archives/cpi_10152010.htm"),
    ("2012-09", "2012-10-16", "archives/cpi_10162012.htm"),
    ("2014-08", "2014-09-17", "archives/cpi_09172014.htm"),
    ("2014-09", "2014-10-22", "archives/cpi_10222014.htm"),
    ("2016-09", "2016-10-18", "archives/cpi_10182016.htm"),
    ("2018-09", "2018-10-11", "archives/cpi_10112018.htm"),
    ("2020-09", "2020-10-13", "archives/cpi_10132020.htm"),
    ("2022-09", "2022-10-13", "archives/cpi_10132022.htm"),
    ("2024-09", "2024-10-10", "archives/cpi_10102024.htm"),
]


def main():
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    previous_path = HERE / "acquisition.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {"records": []}
    pinned = {
        attempt["file"]: attempt["sha256"]
        for record in previous["records"]
        for attempt in record["attempts"]
        if "file" in attempt
    }
    records = []
    for reference, release, bls_path in RELEASES:
        original = "https://www.bls.gov/news.release/" + bls_path
        mirror = (
            "https://fraser.stlouisfed.org/files/docs/publications/bls/newsreleases/cpi/cpi_"
            + release.replace("-", "")
            + ".pdf"
        )
        attempts = []
        for url, name, representation in [
            (original, "bls_" + release + Path(bls_path).suffix, "direct_bls_release"),
            (mirror, "cpi_" + release.replace("-", "") + ".pdf", "fraser_archived_bls_release_pdf"),
        ]:
            path = raw / name
            observed = datetime.now(timezone.utc).isoformat()
            try:
                if path.exists():
                    data = path.read_bytes()
                    expected = pinned.get("raw/" + name)
                    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
                        raise ValueError("existing artifact digest differs from prior acquisition")
                    status = "existing_bytes"
                else:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "Ballotmancy historical research contact via repository"}
                    )
                    with urllib.request.urlopen(request, timeout=25) as response:
                        data = response.read()
                    if representation.endswith("_pdf") and not data.startswith(b"%PDF-"):
                        raise ValueError("not PDF bytes")
                    path.write_bytes(data)
                    status = "downloaded"
                attempts.append(
                    dict(
                        url=url,
                        status=status,
                        observed_at=observed,
                        representation=representation,
                        file="raw/" + name,
                        sha256=hashlib.sha256(data).hexdigest(),
                        bytes=len(data),
                    )
                )
            except (urllib.error.URLError, ValueError) as error:
                attempts.append(
                    dict(
                        url=url,
                        status="fetch_failed",
                        observed_at=observed,
                        error=str(error),
                        representation=representation,
                    )
                )
        records.append(dict(reference_month=reference, release_date=release, original_url=original, attempts=attempts))
        print(reference, [(item["representation"], item["status"]) for item in attempts], flush=True)
    if previous_path.exists():
        with (HERE / "acquisition-history.jsonl").open("a") as history:
            history.write(json.dumps(previous, sort_keys=True) + "\n")
    (HERE / "acquisition.json").write_text(
        json.dumps(dict(schema="bls_cpi_archive_acquisition_v1", records=records), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
