# tests/fixtures

Static text fixtures consumed by the crosstab parser tests.  Real PDFs and
scraped artifacts live under `data/raw/`; this directory holds **synthetic
extractions** that mirror the format of real pollster output so parser logic
can be tested without pinning large binary blobs into the test tree.

## quantus_ga_senate_2025_09.txt

Synthetic plain-text crosstab modeled on Quantus Insights' September 9–12,
2025 Georgia U.S. Senate poll (N=624 likely voters).  The topline numbers
(Ossoff 40 / Collins 37) match the entry in `data/polls/polls_2026.csv`; the
demographic breakdowns are plausible but illustrative — their purpose is to
exercise `scripts/parse_quantus_report.py`, not to make claims about the
real Quantus crosstab distribution.

Swap in real numbers once a Quantus PDF or HTML crosstab is saved to
`data/raw/quantus/`.  The parser itself does not need to change — it reads
text from any source.
