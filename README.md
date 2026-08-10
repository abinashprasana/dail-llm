<div align="center">

# 🏛️ Dáil LLM — Irish Parliamentary Transformer

**A character-level language model trained from scratch on nearly a century of Irish parliamentary debate.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://abinashprasana-dail-llm-dail-llmappstreamlit-app-um65t7.streamlit.app/)
[![Perplexity](https://img.shields.io/badge/Perplexity-4.07-2ea44f?style=for-the-badge)](.)
[![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge)](.)

<br/>

*Dáil Éireann 1919–2013 · Harvard Dataverse · 4.4M speeches · Built on CPU · No pretrained weights*

</div>

---

## 📖 What This Project Is

I built this project to understand how transformers actually work, not just in theory but from the ground up. Instead of using a pretrained model or someone else's tokenizer, I wrote everything from scratch in PyTorch and trained it on real speeches from the Dáil Éireann, the lower house of the Irish parliament. The Dáil has been sitting since January 1919, and the dataset records every speech made by every elected TD across nearly a century of Irish legislative history.

The model learns to generate text that looks like parliamentary debate, one character at a time, with no pretrained weights and no external APIs. Everything runs locally on CPU.

The project also includes a four tab Streamlit dashboard where you can type a prompt and watch the model generate text, explore how attention weights flow across characters as a heatmap, and review the full evaluation results and training curves.

---

## 🎬 Live Demo

[![Open Live App](https://img.shields.io/badge/Open%20Live%20App%20%F0%9F%9A%80-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://abinashprasana-dail-llm-dail-llmappstreamlit-app-um65t7.streamlit.app/)

The app is deployed and running on Streamlit Cloud. No setup needed. You can generate parliamentary text from a seed prompt, visualise attention weights as a heatmap across all 4 layers and 8 heads, and browse the full evaluation results and training curves.

---

## ⚡ Quick Stats

<div align="center">

|  | 📉 Test Perplexity | 📝 Corpus BLEU | 🔁 Repetition Score | 🔢 Parameters | 🔄 Training Steps |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Score** | **4.07** | **0.0104** | **0.0000** | **~3.26M** | **2,000** |

</div>

---

## 🗃️ Dataset

<div align="center">

| Detail | Value |
|:---|:---|
| 📚 Name | Dáil Éireann Parliamentary Debates 1919–2013 |
| 🌐 Source | Harvard Dataverse |
| 🗣️ Total speeches | 4,443,713 |
| 👥 Total speakers (TDs) | 1,178 |
| 📅 Full date range | January 1919 to 2013 |
| 📅 Date range used | 1950 to 2013 |
| 🔤 Language filter | English only |
| 📦 Subset extracted | 6.1 MB of clean text |
| ✂️ Train / Val / Test split | 90% / 5% / 5% |
| 🔧 Encoding repair | ftfy library |

</div>

Only speeches from 1950 onwards were used because earlier debates contain a higher proportion of Irish language content. A non-ASCII character filter removed the remaining Irish language speeches. The full 3.44 GB dataset file is not included in this repository and you can access it directly from Harvard Dataverse using the citation below.

> **Citation:** Proksch, S.O. and Slapin, J.B. (2010). *Database of Parliamentary Speeches in Ireland, 1919–2013*. Harvard Dataverse. https://doi.org/10.7910/DVN/6MZN76

---

## 🧠 Model Architecture

The model is called `DailTransformerLM` and is a decoder-only transformer built entirely from scratch in PyTorch with no pretrained components.

```mermaid
flowchart TD
    classDef io      fill:#1d4ed8,color:#fff,stroke:#1e40af,rx:8
    classDef embed   fill:#4f46e5,color:#fff,stroke:#4338ca,rx:8
    classDef block   fill:#7c3aed,color:#fff,stroke:#6d28d9,rx:8
    classDef attn    fill:#6d28d9,color:#fff,stroke:#5b21b6,rx:8
    classDef ff      fill:#9333ea,color:#fff,stroke:#7e22ce,rx:8
    classDef head    fill:#065f46,color:#fff,stroke:#064e3b,rx:8

    A["🔤  Input Characters\nraw text prompt"]:::io
    B["Token Embedding\n256-dim lookup table"]:::embed
    C["Positional Embedding\n256-dim · learned"]:::embed
    D["➕  Add Embeddings\n+ Dropout 0.1"]:::embed

    A --> B
    A --> C
    B --> D
    C --> D

    D --> BLK

    subgraph BLK["🔄  Transformer Block  ×4"]
        direction TB
        LN1["LayerNorm"]:::block
        MHA["Multi-Head Self-Attention\n8 heads  ·  32-dim per head\ncausal lower-triangular mask"]:::attn
        R1["➕  Residual connection"]:::block
        LN2["LayerNorm"]:::block
        FFN["Feed-Forward Network\n256 → 1024 → 256\nGELU · Dropout 0.1"]:::ff
        R2["➕  Residual connection"]:::block
        LN1 --> MHA --> R1 --> LN2 --> FFN --> R2
    end

    BLK --> FLN["Final LayerNorm"]:::embed
    FLN --> LIN["Linear Projection\nvocab size ≈ 90 characters"]:::head
    LIN --> OUT["🔤  Next Character\npredicted token"]:::io
```

<div align="center">

| Component | Value |
|:---|:---|
| 🏗️ Model type | Character-level decoder-only transformer |
| 📚 Transformer layers | 4 |
| 👁️ Attention heads | 8 |
| 📐 Embedding dimension | 256 |
| 🔲 Head dimension | 32 |
| 🪟 Context window | 256 characters |
| ⚡ Feed-forward expansion | 4× (256 → 1024 → 256) |
| 🔥 Activation | GELU |
| 💧 Dropout | 0.1 |
| 🔡 Tokeniser | Character-level, vocabulary built from training data |
| 📍 Positional encoding | Learned embeddings |
| 🔒 Attention | Causal autoregressive (lower-triangular mask) |
| 🔢 Total parameters | Approximately 3.26 million |
| ⚙️ Optimiser | AdamW, learning rate 3e-4 |
| 🔄 Training steps | 2,000 |
| 📦 Batch size | 32 |
| 📊 Eval frequency | Every 200 steps |

</div>

---

## 📊 Evaluation Results

<div align="center">

| Metric | Score | What It Means |
|:---|:---:|:---|
| 📉 Perplexity | **4.07** | The model narrows each next character down to roughly 4 equally plausible options. A random guess over an 80-character vocabulary would score around 80. |
| 📝 Corpus BLEU | **0.0104** | Measures exact word-level overlap with held-out text. Near zero is expected for a character-level model generating novel sequences. |
| 🔁 Repetition Score | **0.0000** | No repeated 3-gram patterns were found in any of the five generated samples. The model does not loop. |

</div>

---

## 💬 Generated Samples

> All samples generated using `model_best.pt` at temperature 0.8 with 200 new tokens.

<details>
<summary>🎙️ Prompt: "The Minister for"</summary>

```
The Minister for Finance whether he is aware should be not agreements.
I have not fraid the country that the necessity has been seend to pay
the Deputy. That is not in the next of the committee of the workers.
```
</details>

<details>
<summary>🎙️ Prompt: "In this House"</summary>

```
In this House would avail of midle advise the could private of the banks
that had normal in the reasonable of by the Government are compulsorilising
the pools now to make the proposals of the regarding during the
```
</details>

<details>
<summary>🎙️ Prompt: "The question before us"</summary>

```
The question before us to do that powers to receive the milk of simple
and the point of view he is the problem of it shapping this producer for
a productive (either to conluct that price? Then. There is not the Minister fo
```
</details>

<details>
<summary>🎙️ Prompt: "I wish to raise"</summary>

```
I wish to raise that it would, are work so to cappital a matter with the
speed that us doone that they would have been an urgoning to do mine in
borrow that is concesssion the matter as that it was not a making in a
```
</details>

<details>
<summary>🎙️ Prompt: "On the matter of"</summary>

```
On the matter of the third total of Industry and Commerce. There is no
enough agreement what are possibly but to operate would be donel this
schedule to because the price of the possibilities, there are in told it is
```
</details>

<br/>

> The outputs are not grammatically correct English and that is completely expected. What the model actually learned is parliamentary register. Words like Minister, Deputy, Government, Finance, and Industry and Commerce appear in roughly the right positions. Punctuation is placed with approximate correctness. The character sequences feel plausible rather than random noise, which is honestly quite impressive for 3.26 million parameters trained on 6.1 MB of text.

---

## 📁 Project Structure

```
dail-llm/
├── 📄 config.py                      all hyperparameters and paths
├── 🚀 train_pipeline.py              single entry point for the full pipeline
├── 📋 requirements.txt               dependencies
├── 🔧 pyproject.toml                 build configuration
│
├── 📦 dail_llm/
│   ├── 🧩 inference.py               ModelWrapper class for clean model loading
│   ├── 📂 data/
│   │   ├── extract_dail.py           extracts and cleans speeches from the 3.44 GB dataset
│   │   ├── dataset_builder.py        creates train, validation and test splits plus RAG chunks
│   │   └── tokenizer.py              character-level tokenizer
│   ├── 📂 model/
│   │   ├── transformer.py            full model architecture
│   │   ├── train.py                  training loop with checkpointing
│   │   └── generate.py               command line text generation
│   ├── 📂 eval/
│   │   ├── metrics.py                perplexity, BLEU and repetition functions
│   │   └── evaluate.py               evaluation runner
│   ├── 📂 visualisation/
│   │   ├── attention_viz.py          attention heatmap generation
│   │   └── training_plots.py         loss and perplexity curve plots
│   ├── 📂 rag/
│   │   └── retriever.py              TF-IDF retrieval over SQLite document store
│   └── 📂 app/
│       └── streamlit_app.py          four tab Streamlit dashboard
│
└── 📂 outputs/
    ├── checkpoints/                  trained model weights
    ├── plots/                        loss curves and perplexity plots
    └── evaluation_results.md         full evaluation report
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/abinashprasana/dail-llm.git
cd dail-llm
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download `Dail_debates_1919-2013.tab` from Harvard Dataverse using the citation link above. Place the file in the `dataverse_files/` folder.

**4. Run the full pipeline**
```bash
python train_pipeline.py
```

This runs all four steps in order: extract, split, train, evaluate. If `dail_debates_clean.txt` already exists the extraction step is skipped automatically.

**5. Launch the Streamlit dashboard**
```bash
streamlit run dail_llm/app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

<details>
<summary>⚙️ Run individual steps</summary>

```bash
# Step 1: Extract clean text from the full dataset
python -m dail_llm.data.extract_dail

# Step 2: Create train, validation and test splits
python -m dail_llm.data.dataset_builder

# Step 3: Train the model
python -m dail_llm.model.train

# Step 4: Run evaluation
python -m dail_llm.eval.evaluate

# Generate text from the command line
python -m dail_llm.model.generate --prompt "The Minister for" --max_new_tokens 300 --temperature 0.8
```
</details>

---

## ⚠️ Limitations

This is a small educational project, not a production language model. A few things worth knowing before drawing conclusions from the outputs.

The model operates on individual characters rather than words. It has no concept of what a word is, which makes grammatical coherence difficult to achieve. The context window of 256 characters covers roughly 40 to 50 words, so the model forgets the beginning of a long sentence before it finishes it. Training was done on 6.1 MB of text, which is a small amount by modern standards, and at 3.26 million parameters the model has a fraction of the capacity of even the smallest publicly available language models.

Despite these constraints, the model learned something real. It produces parliamentary vocabulary in roughly appropriate positions, uses punctuation with approximate correctness, and generates novel sequences without looping.

<div align="center">

| 🔧 Possible Improvement | 📈 Expected Effect |
|:---|:---|
| BPE or WordPiece tokeniser | 256 tokens would cover roughly 150 words instead of 40 |
| Larger model (8 layers, 512 dim) | More capacity for pattern generalisation |
| More training data (100 MB) | Broader vocabulary and phrase exposure |
| GPU training | 50 to 100 times faster, enabling longer runs |
| Fine-tuning GPT-2 on this corpus | Start from a model that already knows English |

</div>

---

## 👤 Author

**Abinash Prasana Selvanathan**

*If you found this useful, feel free to ⭐ star the repo.*
