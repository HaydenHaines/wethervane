# Earlier-vintage Senate polling source review

## Finding

I did not find a public, dated pre-October 23, 2018 Senate-poll dataset that has all of the requested properties together: stable poll, question, and sample identifiers; original release links and dates; and earlier-cycle poll observations with election outcomes.

The strongest pre-cutoff archive found is FiveThirtyEight's May 30, 2018 `pollster-ratings/raw-polls.csv`. FiveThirtyEight's article of that date says its downloadable pollster-ratings database covered congressional and gubernatorial polls in the final 21 days since 1998, with more than 8,500 polls overall. The pinned GitHub commit has a valid signature according to GitHub's API, and the existing local copy matches its tree blob. Its original 2018 README says `pollno` is a FiveThirtyEight poll ID; it also defines `polldate` as median field date and warns that `samplesize` may be estimated. The file has candidate toplines and actual election outcomes, but it lacks question IDs, source sample IDs, question wording, population, release date, and original source URL. It is a promising dated source candidate for a 2018 replay using only completed 2012/2016 cycles, subject to source and overlap review; it is not a poll-question/sample archive.

The later FiveThirtyEight `polls` documentation describes the richer schema (poll ID, question ID, population, field interval, `created_at`, URL) but that documentation first entered the public repository on January 29, 2019, after the requested cutoff. Its current download link has been redirected to ABC News; direct current endpoint and raw-repository file requests did not return the dataset. This richer source cannot establish what was available by October 23, 2018. The 2018-era dataset and the post-2018 detailed polls data must remain separate vintages.

## Concrete identity bridge and unresolved Tierney row

An original Rasmussen release provides a useful example of what an event-level bridge can and cannot establish. Rasmussen's October 25, 2012 release and linked question page identify a single-day October 21 Arizona likely-voter Flake/Carmona Senate survey, N=500, with results 50% Flake, 44% Carmona, 3% someone else, and 3% undecided. The May 2018 FiveThirtyEight archive row `5385932` records the same pollster, Arizona Senate race, median field date, sample size, and D/R toplines. This is a verified primary-release-to-538 event correspondence.

The prepared Tierney row `tierney_cnn:row:3318` has the same state/cycle/date/pollster label and author-derived candidate values 220/250, which yield the same conditional D/R share. It has no source ID, field start, population, question, publication date, or original release lineage; its derived values sum to 470 rather than the poll's N=500. Thus the 538-to-Rasmussen link does not authenticate the Tierney row's sample or instrument. The Tierney relation remains a strong but unverified candidate, as documented in the preserved earlier packet.

## Other sources screened

- HuffPost Pollster's 2017 API announcement described a question-oriented API and raw data. The service was live and editorially mutable at the time; its announcement explicitly says returned values can change when editors edit the database. Direct requests to its former endpoint now return 404. I found no retained, dated, complete pre-cutoff API response or authenticated dataset vintage in this pass.
- The 2017 AAPOR report says its poll-level input sources were FiveThirtyEight, HuffPost Pollster, and RealClearPolitics, and that design information was supplemented manually from releases, stories, methodology reports, and pollster sites. The public report is a valuable methodological and partial documentation bridge, not a source-ID complete poll-question dataset or a downloadable full event crosswalk.
- Yehu Chen's repository snapshots and Tierney's replication files in the prior packets postdate the 2018 cutoff and lack stable poll/question/sample IDs and row-to-release crosswalks. They cannot supply a pre-2018 data vintage.

## Reproducible source references and limits

Exact URLs, response status, hashes, schema observations, and prior local file paths are in `evidence.json`. Only the two-byte empty GitHub API response for the historical `polls/README.md` path query is captured locally here; raw polling data were not copied. Existing 2018 FiveThirtyEight and Rasmussen source captures remain in their prior directories, unmodified. No fits, row admission, or owner data changes were made.
