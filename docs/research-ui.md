# Research interface

Dáil LLM supports reproducible training, speech-memory inspection, and evaluation across historical periods. The Research page presents the completed CPU pilot separately from the checkpoint used by the model lab.

The page has four views: Overview, Speech memory, Historical evaluation, and Methods. Each view can be linked directly with `?view=overview`, `?view=memory`, `?view=history`, or `?view=methods`. Existing evaluation plots keep their `/research/*.png` addresses.

## Recorded examples

The checked-in release at `frontend/public/research-data/pilot` works without a live research backend. It includes eight prefixes and all three equal-sized memory policies. The first prefix is `The Minister for`; the other seven are selected deterministically from earlier-test speeches. Phrase and continuation labels are heuristic. Examples are not chosen by the size of an improvement.

Each example includes original probabilities and a new search after excluding each contributing speech, one at a time. The interface checks file hashes before displaying the example. Removing a speech excludes all its entries, including entries that were not among the original neighbours. Probability bars retain their original scale and report the undisplayed probability mass.

Source excerpts are limited to 96 characters around a stored target. The highlighted character is the target at the recorded normalized-text offset. A source-removal effect is evidence about retrieval, not training-data attribution or factual accuracy.

Execution dates and date-stamped directory names are not included in reader-facing copy or downloads. Dates of speeches, corpus periods, and historical events remain part of the method. Internal execution records retain their provenance.

## Export a verified release

Install the optional research dependencies, then use an existing completed run. The destination must be empty and separate from the source run:

```bash
python -m pip install -e ".[research]"
python -m dail_llm.research.publication --run outputs/research/my-run --destination outputs/research/public-release
```

The export checks the report, verification result, evaluation hashes, checkpoint hashes, tokenizer, corpus, and all three memories. It does not train, tune, or update the original run. Review the exported files, then copy that release into `frontend/public/research-data/pilot` and rebuild the frontend. Keep every example file with the corresponding summary.

Only the current CPU-pilot presentation is included. A larger study needs its own interpretation and reviewed copy before publication; replacing the files alone is not a way to publish new claims.

## Optional live inspection

Create a separate private runtime bundle:

```bash
python -m dail_llm.research.publication --run outputs/research/my-run --destination outputs/research/private-runtime --private-bundle
```

This copies the prepared records, validation artifacts, required best checkpoints, and selected memories. It excludes the original archive, latest checkpoints, review answer key, and unrelated run files. Prepared speeches and weights remain private. Never put this directory under the frontend or another public static directory.

For a local server, set `DAIL_RESEARCH_RUN` to this directory. `DAIL_RESEARCH_PUBLIC` defaults to `frontend/dist/research-data/pilot`; set it explicitly to `frontend/public/research-data/pilot` while using the Vite development server.

```bash
docker build --build-arg INSTALL_RESEARCH=true -t dail-llm:research .
docker run --rm -p 8000:8000 \
  --mount type=bind,source=/absolute/path/private-runtime,target=/research-private,readonly \
  -e DAIL_RESEARCH_RUN=/research-private dail-llm:research
```

The standard image leaves live research disabled and serves the recorded examples. Local research runs are excluded from the Docker context, and served artifacts are copied explicitly. The research image adds optional dependencies; it does not copy the private runtime bundle into the image.

`GET /api/v1/research/capabilities` reports the release identity and live availability. `POST /api/v1/research/inspect` accepts:

```json
{
  "release_id": "the-release-id-from-capabilities",
  "prefix": "The Minister for",
  "policy": "uniform",
  "excluded_speech": null
}
```

Policies are `uniform`, `speech_balanced`, and `context_diverse`. Prefixes contain 1–256 Unicode characters. Unsupported characters retain their positions and map to the research tokenizer's explicit unknown token. The optional excluded speech must belong to the selected memory. Checkpoint and retrieval settings are fixed by the verified release.

The service caches validated assets and uses request-local search state. It shares the model lab's inference gate and rate limit. A mismatched public/private release disables live inspection. Requests return 409 for a stale release, 422 for invalid input, 429 for capacity/rate limits, and 503 when live inspection is unavailable. Custom prefixes are not stored by the UI or included in request URLs.

## Reading the evidence

The pilot uses one seed and 100 training steps per decoder. The five-gram baseline performs better than the pilot transformers. Both memory-policy intervals relative to uniform selection cross zero. Historical matching yields one pair, leaving 46 earlier and 34 later speeches unmatched. These limits remain beside the results.

The five-gram continuing-speaker and matched-passage views aggregate its existing per-speech measurements over the same cohorts used for the decoders. No new fitting is involved. Input masks are sensitivity analyses, not simulations of political behaviour. Recorded affiliations include outdated labels and do not establish dated party membership.

## Build and verification

```bash
python -m pytest -q
python -m ruff check dail_llm tests
cd frontend
pnpm run lint
pnpm run test
pnpm run build
pnpm run test:e2e
```

The production build checks the complete static import graph against the existing homepage budgets. Research and memory-explanation code load separately. It also verifies every public example hash and enforces the JSON payload limits. Browser tests cover the recorded interactions, navigation, accessibility, narrow layouts, and the existing chamber fallbacks. Visual baseline changes must be reviewed rather than accepted wholesale.
