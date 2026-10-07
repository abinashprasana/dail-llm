# Dáil LLM

Explore Dáil Éireann debates through three separate tools: sourced debate search, a character model, and recorded research experiments. Each uses different data and answers a different kind of question.

| Route | What it shows | Evidence |
| --- | --- | --- |
| **Ask the debates** (`/ask`) | Retrieves speaker passages from the Oireachtas Official Report and links to the original debate | A local index of Dáil debate XML from 2014 onward; availability and coverage come from the connected API |
| **Model Lab** (`/lab`) | Continues a prompt one character at a time and displays attention and evaluation results | A 3.27 million parameter model trained on 9,080 speeches dated 15 February–25 April 1950 |
| **Research** (`/research`) | Explores speech memory and historical comparisons | A separate CPU pilot using selected 2008–2011 debates |

[Open the public site](https://dail-llm.vercel.app/) · [Read the interface specification](docs/UI_DESIGN_SPEC.md)

The public site serves the redesigned Home, Model Lab, Research, and `/ask` pages. Its Q&A API currently reports that debate search is unavailable because no public index is connected; the Ask page disables search accordingly. Public Q&A still needs citation review, an independently checked evaluation, and free-tier sizing before launch.

## What the measurements show

The served character model has four decoder layers, eight attention heads, a 256-character context, and no pretrained weights. Its saved held-out evaluation reports **4.07 perplexity**, **2.024 bits per character**, and **58.67% next-character accuracy**. These measure character prediction, not factual answering. See [the evaluation JSON](outputs/evaluation_results.json) for the checkpoint settings, seed, and generated samples.

The separate Research pilot used a short CPU training budget. On its earlier test split, a Witten–Bell five-gram scored **1.8572 bits per character**, while the unconditioned decoder scored **3.5646**; lower is better. The speech-memory comparisons did not establish a clear gain, and the historical matched comparison contained only one pair. These pilot scores use different data and accounting from the served checkpoint, so they should not be compared directly with its 2.024 figure. [Results and limits](docs/pilot-results.md) · [Research reproduction guide](docs/research.md)

The character model can generate broken or incorrect text. The Q&A tool retrieves attributed passages but does not determine whether a speaker's claims are true. When answer generation is unavailable, it returns cited excerpts. The [Q&A evaluation record](eval/README.md) explains why the current exploratory questions do not yet support a public launch.

## Run the application

You need Python **3.12 or newer** and Node.js with pnpm. The repository includes the served checkpoint, saved evaluation, and recorded Research examples. You do **not** need to download the historical archive just to browse the Lab and Research pages.

From the repository root:

```powershell
python -m pip install -e ".[research]"
cd frontend
pnpm install --frozen-lockfile
pnpm run build
cd ..
python -m dail_llm.api
```

Open `http://127.0.0.1:8000`. The compiled frontend is served by FastAPI. The Research page uses its recorded examples without a private research backend. For frontend development, run `pnpm run dev` in `frontend/` and use the Vite preview at `http://127.0.0.1:5173`; its `/api` requests proxy to port 8000.

Docker can build and serve the same application with `docker compose up --build`. Optional runtime settings are listed in [`.env.example`](.env.example). The private Research inspection bundle has its own [setup guide](docs/research-ui.md).

### Reproduce the 1950 model

Download `Dail_debates_1919-2013.tab` from the [Harvard Dataverse record](https://doi.org/10.7910/DVN/6MZN76) and place it in `dataverse_files/`. Then run:

```powershell
python train_pipeline.py
```

This extracts the 1950 subset, builds the splits, trains, and evaluates. It skips extraction if `dataverse_files/dail_debates_clean.txt` already exists. Training takes substantially longer than opening the included checkpoint. The configuration is in [`dail_llm/config.py`](dail_llm/config.py); the resulting corpus manifest and evaluation are in [`outputs/`](outputs/).

The extractor keeps speeches of at least 50 characters and excludes speeches with more than 40% non-ASCII characters. That threshold is **not** a language detector. The original archive covers 1919–2013, while this checkpoint uses only ten weeks in 1950. The repository's [dataset manifest](outputs/dataset_manifest.json) records the actual selected dates, speech count, and source hash.

### Build the current-debate index

Install the Q&A dependencies, download Official Report XML, and build the local search index:

```powershell
python -m pip install -e ".[qa]"
python -m scripts.download_debates --start 2014-01-01
python -m dail_llm.qa.index
```

The downloader saves XML and a SHA-256 manifest under ignored `data/oireachtas/`; reruns reuse unchanged records. Use `--refresh` with a bounded date range to check for revised XML. The SQLite FTS index is `data/oireachtas/passages.sqlite`. Start the API again and `/ask` will report the coverage of that connected index. The [data card](docs/DATA_CARD.md) records one measured local snapshot, its gaps, and the source-link rules. Separately published parliamentary questions are outside this debate index.

The [Official Report](https://www.oireachtas.ie/en/debates/) is the source for the newer debates. Its [open-data licence](https://www.oireachtas.ie/en/open-data/license/) applies Creative Commons Attribution 4.0 terms. The 1919–2013 archive is **Alexander Herzog and Slava J. Mikhaylov, _Database of Parliamentary Speeches in Ireland, 1919–2013_, Harvard Dataverse**, [DOI 10.7910/DVN/6MZN76](https://doi.org/10.7910/DVN/6MZN76).

## Checks and further detail

```powershell
python -m pip install -e ".[dev,research,qa]"
python -m ruff check dail_llm tests scripts
python -m pytest -q -m "not integration"
cd frontend
pnpm run lint
pnpm run test
pnpm run build
pnpm run test:e2e
```

The browser suite uses mocked Q&A responses and covers responsive layouts, keyboard use, reduced motion, and the Research views. These checks do not prove that the public Q&A service is deployed. The FastAPI endpoint list is available at `/api/docs` when the service is running. It includes generation, attention, Research, and the local `/api/v1/qa/capabilities` and `/api/v1/qa/ask` endpoints.

The [repository audit](docs/AUDIT.md) tracks model, hosting, dependency, and verification status. The [infrastructure notes](infra/README.md) describe the free Cloudflare D1 pilot: **30 passages** were imported to test search, links, and Irish fadas. That pilot is not the full corpus. Terraform describes proposed free-tier resources; it has not been applied. Langfuse tracing and LangSmith experiments still require credentials and reviewed evaluation data.

Raw downloads, generated indices, and research run directories are ignored by Git. The recorded Research examples are checked in so that page remains inspectable without a live research service.

**Author:** Abinash Prasana Selvanathan
