# Offline Dáil language-model studies

Dáil LLM supports reproducible training, speech-memory inspection, and evaluation across historical periods.

The [completed CPU pilot](pilot-results.md) records the first results, verification scope, and limitations.

The [Research interface guide](research-ui.md) explains public examples, optional live inspection, and deployment.

The research commands operate in a separate run directory. They do not publish a model, update the web interface, or replace the application's checkpoint and evaluation files.

## Run the CPU pilot

Use Python 3.12 or later and install the optional dependencies with `python -m pip install -e ".[research,dev]"`. Place the original archive at `dataverse_files/Dail_debates_1919-2013.tab`, or pass its location with `--source` when preparing data.

Run these commands from the repository root:

```powershell
python -m dail_llm.research prepare --profile experiments/pilot.toml --run outputs/research/pilot
python -m dail_llm.research train --run outputs/research/pilot
python -m dail_llm.research build-memory --run outputs/research/pilot
python -m dail_llm.research evaluate --run outputs/research/pilot
python -m dail_llm.research.verify --run outputs/research/pilot
python -m dail_llm.research report --run outputs/research/pilot
```

`dail-research` exposes the same commands after installation. A nonempty run directory is rejected by `prepare`. A training directory also requires explicit `--resume` before it can be reused. For example:

```powershell
python -m dail_llm.research train --run outputs/research/pilot --mode none --seed 42 --resume
```

`build-memory` and `evaluate` also accept `--resume`. Memory construction deterministically rebuilds the selected arrays; evaluation skips verified completed results. Resume rejects changes to the model, data, or research implementation. A run lock prevents concurrent commands from writing to the same directory. If a process is forcibly terminated, check that it has stopped before removing its stale `.execution.lock`.

The pilot trains each of the three models for 100 steps, using seed 42 and batches of eight. It selects up to 500,000 training characters and 25,000 characters per evaluation partition, then compares memories of up to 5,000 entries. Whole debate groups can make the actual counts smaller. All models share those records.

Execution time accumulates across commands in `status.json`; time between commands is not charged. The limit is two hours and the process-memory budget is 4 GiB. Budget checks run between bounded batches, so these are cooperative limits, not an operating-system sandbox. Training saves its latest state on a budget interruption. Incomplete stages and missing results remain explicit in the report. Reporting is available after a budget interruption and does not fabricate missing measurements.

For the full study, repeat the command sequence in a new run directory using `experiments/full.toml`. That profile uses seeds 17, 42, and 73, 2,000 steps, batches of 32, up to six million training characters, 300,000 characters per evaluation partition, and 100,000 memory entries per policy. It has no wall-time limit and is never launched automatically. The implementation runs on CPU; GPU acceleration is not required.

## Data and historical comparison

The source is [Herzog and Mikhaylov's parliamentary corpus](https://arxiv.org/abs/1708.04557). Its codebook specifies a ten-column tab-separated file with literal quotation marks. The reader validates that format, retains Unicode, records malformed lines, and preserves original speech text alongside normalized text. It does not classify the language of a passage.

| Partition | Dates |
|---|---|
| Training | January 2008–December 2009 |
| Validation | January–June 2010 |
| Earlier test | July–December 2010 |
| Later test | April–September 2011 |

Selection cycles through months in deterministic order and takes whole groups identified by date and normalized debate title. Oversized groups are counted and skipped. This bounded selection can favour shorter debates, particularly in the pilot; the manifest records the resulting coverage. Exact and near-duplicate removal operates across the selected records, retaining the earliest speech. Near duplicates require at least 50 words and word-five-gram Jaccard similarity of at least 0.9. This is not an exhaustive duplicate catalogue of the entire archive.

The later test follows the [9 March 2011 government transition](https://www.oireachtas.ie/en/debates/debate/dail/2011-03-09/4/). Role labels map Fianna Fáil, Fine Gael, and Labour only. Other affiliations and the transition day are unmapped. These are party-level government/opposition labels, not individual ministerial appointments or claims about parliamentary support agreements.

The three models use the same decoder backbone. The conditioned versions add a learned party or role embedding to each input position. Reports include their parameter counts. Results cover all speeches, continuing speakers, the mapped role cohort, and one-to-one passages matched within speaker by training-fitted TF-IDF similarity and length. A small or unmatched cohort is evidence of limited coverage, not evidence of no historical effect.

Party conditioning uses the archive's recorded affiliation. Those labels are not a verified history of individual party membership: the selected material includes Eamon Gilmore under The Workers' Party in 2008. Such labels remain preserved and unmapped for the role comparison. The three-party role mapping is therefore a restricted cohort based on recorded affiliations; conclusions require checking those affiliations against dated membership records. The archive's label “The Labour Party” is recognized explicitly.

This comparison adapts [Hirst et al.'s study of institutional-role confounding](https://www.benjamins.com/catalog/dapsac.55.05hir) and [Rheault and Cochrane's metadata-based parliamentary models](https://doi.org/10.1017/pan.2019.26). One transition cannot separate every change in topic, personnel, and political context.

## Memory and source inspection

Following [kNN-LM](https://arxiv.org/abs/1911.00172), memory interpolates a nearest-neighbour next-character distribution with the decoder distribution. [Adaptive Semiparametric Language Models](https://aclanthology.org/2021.tacl-1.22/) provides related character-level precedent. This implementation uses final-layer-normalized representations and exact squared-Euclidean search.

Uniform sampling is compared with speech-balanced sampling and speech-balanced sampling capped at four entries per normalized trailing-64-character context. All policies use the smallest achieved entry count. The checkpoint, tokenizer, memory arrays, entries, and corpus are bound by hashes.

Validation selects neighbours, interpolation weight, and retrieval temperature. Test scores never select those values. Weight zero is an eligible outcome: retrieval may provide no useful gain.

After evaluation:

```powershell
python -m dail_llm.research inspect --run outputs/research/pilot --seed 42 --policy context_diverse --prefix "The Minister for"
python -m dail_llm.research inspect --run outputs/research/pilot --seed 42 --policy context_diverse --prefix "The Minister for" --exclude SPEECH_ID
```

Replace `SPEECH_ID` with an ID from the first output. Repeat `--exclude` to exclude additional speeches. The command removes every matching entry and searches again. It measures a contribution through retrieval for a fixed prefix. It does not identify what caused a trained weight or verify a generated statement.

## Reading the evidence

The research tokenizer reserves explicit unknown and padding IDs. Its character vocabulary is fitted on training data. Every character in normalized text keeps its position during tokenization; memory offsets refer to that normalized text. The original archive text is preserved separately. Supported-target bits per character excludes unknown targets and is always accompanied by coverage; mapped-token loss includes the unknown category and is not the exact likelihood of an unseen character.

Windows are speech-local and nonoverlapping, with one preceding character. Every target except the first character of each speech has exactly one owner, including partial final windows. Context near a window boundary is shorter than 256 characters. All scored methods, including the interpolated Witten–Bell five-gram baseline, follow this convention. Fixed-prefix inspection uses the supplied prefix; it need not reproduce a dataset window unless the same prefix is supplied.

Word, recorded member-name, and recurring-phrase slices address the distinction in [Long-Tail Crisis in Nearest Neighbor Language Models](https://aclanthology.org/2025.findings-naacl.331/). Complete-span accuracy is teacher-forced next-character accuracy across the span, not free-generation success. Slice labels are heuristic and overlapping spans are not independent samples. Generation outputs include repeated trigrams, exact training eight-gram overlap, and throughput; these do not establish readability or factual accuracy.

The report contains paired debate-bootstrap intervals and variation across available training seeds. Pilot intervals are exploratory. `review-pack.json` reserves examples for ordinary continuations, recorded member names, and recurring phrases, up to 120 items in total. Source probes and generated continuations are identified as such; the generating system is kept in the separate `review-answer-key.json`. Stratum counts are reported, including zero where the selected records contain no matching examples. These are heuristic strata awaiting review. Readability and substantive-language conclusions require review. No LLM judge supplies human ratings.

Run artifacts, local speech extracts, and weights are ignored by Git. Reports can be shared after reviewing source terms and the included excerpts. Raw-archive redistribution is not part of this workflow.

## Verification

Run `python -m pytest` for the scientific and compatibility tests. The synthetic end-to-end study runs in CI without the archive. Tests cover unknown characters and padding, independent likelihood calculations, exact CPU resume, brute-force retrieval agreement, source exclusion, temporal roles, input-mask controls, and malformed or incompatible artifacts.

The separate `research.verify` command checks a completed real run. It checks every retained speech's target positions, partition ownership, memory eligibility, equal memory sizes, and checkpoint selection. It independently recomputes sampled speech losses in float64 and compares sampled retrieval distances with brute force. Results and input hashes are written to `verification.json`. This numerical audit does not certify the source metadata or replace the blinded human review.

Local API adversarial tests also exercise bounds, path traversal, CORS, rate limits, and restricted checkpoint loading. Rate limiting uses the client address supplied by the ASGI server. A reverse-proxy deployment must configure the server's trusted proxy addresses; raw forwarded headers are not accepted directly as client identities.
