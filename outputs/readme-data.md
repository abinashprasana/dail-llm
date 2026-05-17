# README Data — Dáil LLM
Raw information compiled for writing the final README.
Source files: evaluation_results.md, config.py, transformer.py, extract_dail.py, evaluate.py

---

## 1. EXACT METRICS

| Metric | Value | Notes |
|--------|-------|-------|
| Perplexity (test set) | 4.07 | Lower is better. exp(avg cross-entropy loss). |
| Corpus BLEU | 0.0104 | Higher is better. Range 0–1. Very low is expected for character-level models. |
| Avg Repetition Score | 0.0000 | Lower is better. Fraction of repeated 3-grams. Zero means no repeated trigrams. |
| Final training loss (implied) | ~1.40 | Derived: ln(4.07) ≈ 1.403. Not saved to text file, only in loss.png. |
| Final validation perplexity | 4.07 | model_best.pt was chosen by lowest val loss. Test perplexity matches this closely. |

### What these numbers mean in plain English
- **Perplexity 4.07**: On average the model is choosing among ~4 equally plausible next characters at each step. For reference, a random guess over an 80-character vocab would give perplexity ~80. A perplexity of 4 means the model has learned real structure.
- **BLEU 0.0104**: Very close to zero. The generated text shares almost no word-level n-grams with held-out test text. Expected for a character-level model trained on 6 MB with no word-level concepts.
- **Repetition 0.0000**: None of the five generated samples repeated any 3-word sequence. The model does not loop.

---

## 2. EXACT MODEL ARCHITECTURE

Source: `config.py` and `dail_llm/model/transformer.py`

| Property | Value |
|----------|-------|
| Model class | DailTransformerLM |
| Transformer layers (N_LAYERS) | 4 |
| Attention heads (N_HEADS) | 8 |
| Embedding dimension (EMBED_DIM) | 256 |
| Head dimension | 32 (= 256 / 8) |
| Context window (BLOCK_SIZE) | 256 characters |
| FFN expansion | 4× (256 → 1024 → 256) |
| Activation | GELU |
| Dropout | 0.1 |
| Tokenizer | Character-level (dynamic vocab from training text) |
| Positional encoding | Learned embeddings (nn.Embedding) |
| Attention masking | Causal (lower-triangular mask — autoregressive) |
| Weight init | Normal(0, 0.02) for Linear and Embedding |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Batch size | 32 |
| Max training steps | 2000 |
| Eval every | 200 steps |

### Estimated parameter count
Calculated with vocab_size ≈ 90 (typical for English text with punctuation and digits):

| Component | Parameters |
|-----------|-----------|
| Token embedding | vocab × 256 ≈ 23,040 |
| Position embedding | 256 × 256 = 65,536 |
| Per block: QKV projection | 3 × 256 × 256 = 196,608 |
| Per block: Output projection | 256 × 256 = 65,536 |
| Per block: FFN (both linear layers) | 2 × 256 × 1024 = 524,288 |
| Per block: LayerNorm (×2) | 2 × 2 × 256 = 1,024 |
| 4 blocks total | 4 × 787,456 = 3,149,824 |
| Final LayerNorm | 512 |
| Output head | vocab × 256 ≈ 23,040 |
| **Total** | **≈ 3.26M parameters** |

Note: The training script prints the exact parameter count at the start of each run (`Model parameters: X.XXM`). The README previously estimated 5–8M which was based on larger assumed vocab; actual count is approximately 3.2M.

---

## 3. EXACT DATASET INFORMATION

### Full citation
Proksch, S.O. and Slapin, J.B. (2010). *Database of Parliamentary Speeches in Ireland, 1919–2013*. Harvard Dataverse. doi:10.7910/DVN/6MZN76

### Dataset summary
| Property | Value |
|----------|-------|
| Full dataset name | Dáil Éireann Parliamentary Debates 1919–2013 |
| Source | Harvard Dataverse |
| Format | Tab-Separated Values (.tab) |
| Full file size | 3.44 GB (Dail_debates_1919-2013.tab) |
| Total speeches in full dataset | 4,443,713 |
| Total TDs (speakers) in full dataset | 1,178 |
| Full date range | January 1919 – 2013 |
| Date range used in this project | 1950–2013 (post-1950 filter applied) |
| Language filter | English only — speeches with >40% non-ASCII characters discarded (Irish-language proxy) |
| Encoding repair | `ftfy` library (Fixes Text For You) — repairs mojibake and broken Unicode |
| Minimum speech length | 50 characters (shorter speeches discarded) |
| Extracted text file | dataverse_files/dail_debates_clean.txt |
| Extracted text size | ~6 MB (6,334,226 bytes approx.) |
| Target stop condition | Extraction halts once 6 MB of clean text collected |

### What is the Dáil
The Dáil Éireann is the lower house of the Irish parliament (Oireachtas). It has been in session since January 1919. The dataset records every speech made by every TD (Teachta Dála — elected representative) across nearly a century of Irish legislative history. Topics range from independence-era policy debates to post-Celtic Tiger austerity.

### Why only 6 MB from 3.44 GB
The character-level transformer is extremely small (3.2M parameters). Training on the full 3.44 GB dataset would take weeks on CPU and the model would underfit massively because it has far too few parameters to represent that much data. 6 MB gives enough text to learn character-level patterns while keeping training time to 30–60 minutes on a modern laptop CPU.

---

## 4. ALL 5 GENERATED TEXT SAMPLES

Source: `outputs/evaluation_results.md`
Checkpoint used: `model_best.pt`
Generation settings: max_new_tokens=200, temperature=0.8

---

**Sample 1**
Prompt: `"The Minister for"`
Repetition score: 0.0000

```
The Minister for Finance whether he is aware should be not agreements. I have not fraid the country that the necessity has been seend to pay the Deputy. That is not in the next of the committee of the workers. The Mi
```

---

**Sample 2**
Prompt: `"In this House"`
Repetition score: 0.0000

```
In this House would avail of midle advise the could private of the banks that had normal in the reasonable of by the Government are compulsorilising the pools now to make the proposals of the regarding during the
```

---

**Sample 3**
Prompt: `"The question before us"`
Repetition score: 0.0000

```
The question before us to do that powers to receive the milk of simple and the point of view he is the problem of it shapping this producer for a productive (either to conluct that price? Then. There is not the Minister fo
```

---

**Sample 4**
Prompt: `"I wish to raise"`
Repetition score: 0.0000

```
I wish to raise that it would, are work so to cappital a matter with the speed that us doone that they would have been an urgoning to do mine in borrow that is concesssion the matter as that it was not a making in a
```

---

**Sample 5**
Prompt: `"On the matter of"`
Repetition score: 0.0000

```
On the matter of the third total of Industry and Commerce. There is no enough agreement what are possibly but to operate would be donel this schedule to because the price of the possibilities, there are in told it is
```

---

## 5. HOW TO RUN — EXACT COMMANDS

### Prerequisites
Python 3.8 or later. All dependencies installed. The full 3.44 GB dataset file placed in `dataverse_files/`.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Option A — Run everything in one command
```bash
python train_pipeline.py
```
This runs all four steps in sequence: extract → split → train → evaluate. Skips extraction if `dail_debates_clean.txt` already exists.

### Option B — Run steps individually

**Step 1 — Extract the dataset** (skip if dail_debates_clean.txt already exists)
```bash
python -m dail_llm.data.extract_dail
```
Reads `Dail_debates_1919-2013.tab`, applies ftfy, filters English speeches from 1950+, stops at 6 MB.
Output: `dataverse_files/dail_debates_clean.txt`

**Step 2 — Prepare data splits**
```bash
python -m dail_llm.data.dataset_builder
```
Creates 90/5/5 train/val/test split, RAG chunks (JSONL), and SQLite database.
Output: `data/processed/train.txt`, `val.txt`, `test.txt`, `chunks.jsonl`, `data/texts.db`

**Step 3 — Train the model**
```bash
python -m dail_llm.model.train
```
Trains for 2000 steps, saves best checkpoint by validation loss, plots loss and perplexity curves.
Output: `outputs/checkpoints/model_best.pt`, `model_latest.pt`, `model.pt`, `outputs/plots/loss.png`, `val_perplexity.png`

**Step 4 — Evaluate**
```bash
python -m dail_llm.eval.evaluate
```
Runs perplexity, BLEU, repetition on test set. Generates 5 samples.
Output: `outputs/evaluation_results.md`

**Step 5 — Launch Streamlit app**
```bash
streamlit run dail_llm/app/streamlit_app.py
```
Opens a 4-tab dashboard: About, Text Generation, Evaluation Results, Attention Visualisation.

**Generate text from command line**
```bash
python -m dail_llm.model.generate --prompt "The Minister for" --max_new_tokens 300 --temperature 0.8
```

---

## 6. PROJECT STRUCTURE (after reorganisation)

```
dail-llm/
├── config.py                          ← All hyperparameters and paths (project root)
├── train_pipeline.py                  ← Single entry point: runs all 4 steps
├── requirements.txt                   ← pip dependencies
├── README.md                          ← Project readme
├── pyproject.toml                     ← Build config
├── dataset-inspection-report.md       ← Pre-build dataset analysis
├── dail-llm-completion-report.md      ← Post-build status report
│
├── dail_llm/                          ← Main Python package
│   ├── __init__.py
│   ├── config.py                      ← Shim re-exporting from root config.py
│   ├── inference.py                   ← ModelWrapper: load checkpoint + generate
│   ├── app/
│   │   └── streamlit_app.py           ← 4-tab Streamlit dashboard
│   ├── data/
│   │   ├── extract_dail.py            ← Step 1: extract from 3.44 GB .tab
│   │   ├── dataset_builder.py         ← Step 2: train/val/test splits + RAG chunks
│   │   └── tokenizer.py               ← Character-level tokenizer
│   ├── eval/
│   │   ├── evaluate.py                ← Step 4: evaluation runner
│   │   └── metrics.py                 ← Perplexity, BLEU, repetition functions
│   ├── model/
│   │   ├── transformer.py             ← DailTransformerLM architecture
│   │   ├── train.py                   ← Step 3: training loop
│   │   └── generate.py                ← CLI text generation
│   ├── rag/
│   │   └── retriever.py               ← TF-IDF RAG pipeline (SQLite-backed)
│   └── visualisation/
│       ├── attention_viz.py            ← Seaborn attention heatmaps
│       └── training_plots.py           ← Loss + perplexity curve plots
│
├── dataverse_files/
│   ├── dail_debates_clean.txt         ← 6 MB extracted clean text (used by pipeline)
│   ├── Dail_debates_1919-2013.tab     ← 3.44 GB full dataset (not in repo)
│   ├── Dail_debates_1937-2011_ministers.tab  ← Ministerial metadata (unused)
│   └── Codebook.txt                   ← Dataset column documentation
│
├── data/                              ← Generated by pipeline (not in repo)
│   ├── texts.db                       ← SQLite RAG database
│   └── processed/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── dail_clean.txt
│       └── chunks.jsonl
│
├── outputs/
│   ├── checkpoints/
│   │   ├── model_best.pt              ← Best checkpoint by val loss
│   │   ├── model_latest.pt            ← Most recent checkpoint
│   │   ├── model.pt                   ← Copy of best checkpoint
│   │   └── test_ids.pt                ← Encoded test set tensor
│   ├── plots/
│   │   ├── loss.png                   ← Train + val loss curves
│   │   └── val_perplexity.png         ← Validation perplexity over training
│   ├── evaluation_results.md          ← Metrics + generated samples
│   └── analysis-report.md             ← Detailed analysis (separate document)
│
└── legacy/                            ← Pre-rename src/ package (kept for reference)
    ├── config.py                      ← Old config (2 layers, 128 dim — outdated)
    ├── inference.py                   ← Old inference (TinyTransformerLM)
    ├── data_prep/
    ├── model/
    ├── eval/
    └── rag/
```

---

## 7. LIMITATIONS

### What the model cannot do well and why

**1. No word-level understanding**
The model operates on individual characters, not words or subword tokens. It has no internal representation of what a "word" is. It can learn that certain character sequences appear after other sequences, but it has no concept that "Minister" and "minister" are related, or that "Finance" is a department name.

**2. Context window of 256 characters is very short**
256 characters is roughly 40–50 words. A real sentence of parliamentary debate might be 3–4 sentences long before the model has already forgotten the start. It cannot maintain coherent long-range grammatical structure.

**3. 6 MB is a tiny training set**
A modern language model trains on hundreds of gigabytes. 6 MB gives about 6 million characters. The model has seen each character sequence a very limited number of times. Many legitimate parliamentary phrases will appear fewer than 10 times in the training data.

**4. 3.2M parameters is very small**
GPT-2 small has 117M parameters. GPT-3 has 175 billion. At 3.2M parameters, this model simply does not have the capacity to memorise or generalise complex language patterns.

**5. CPU-only training, 2000 steps**
2000 steps with batch size 32 means the model has seen 2000 × 32 × 256 = ~16M characters in training contexts. That is a relatively small number of gradient updates for the complexity of natural language.

**6. BLEU near zero is expected**
BLEU measures exact n-gram overlap. A character-level model inventing plausible-looking text will produce real-looking character sequences that are novel, not copied from the test set. BLEU does not capture this well and is a poor metric for character-level generation.

**7. What the model DID learn**
Despite the limitations, the outputs show real structure:
- The model learned parliamentary register: "The Minister for", "I wish to raise", "the Government are"
- It learned common Irish parliamentary topics: Finance, Industry and Commerce, the Deputy, committees
- It learned punctuation patterns: it uses commas, question marks, and brackets in roughly correct positions
- It does not loop (repetition score 0.0000)
- Perplexity of 4.07 over ~80-char vocab is genuinely non-trivial

### What would improve it

| Change | Expected impact |
|--------|----------------|
| BPE or WordPiece tokenizer | 256-token context = ~150 words instead of 40. Dramatic improvement. |
| Larger model (8 layers, 512 dim) | More capacity to memorise and generalise patterns. Still trainable on CPU with more time. |
| More data (100 MB instead of 6 MB) | Model would see more linguistic variety. |
| GPU training | 50–100× faster training, allowing more steps and larger models. |
| Fine-tune GPT-2 on Dáil corpus | Start from a model that already knows English grammar; fine-tune for Irish parliamentary register. |
| Speaker conditioning | Embed speaker ID as an additional token or prefix. Could learn individual TDs' speaking styles. |
| Better evaluation | Human judgement of coherence, topic relevance, and grammatical acceptability. BLEU and perplexity do not capture these. |
