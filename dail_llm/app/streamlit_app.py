"""
Dáil LLM — Irish Parliamentary Transformer — Streamlit Dashboard

Run from the project root:
    streamlit run dail_llm/app/streamlit_app.py
"""
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from config import (
    PLOTS_DIR, EVAL_RESULTS_PATH,
    N_LAYERS, N_HEADS, EMBED_DIM, BLOCK_SIZE, DATASET_NAME,
)
from dail_llm.inference import ModelWrapper

try:
    from dail_llm.visualisation.attention_viz import visualise_attention, visualise_all_heads
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model weights...")
def get_model():
    return ModelWrapper()


def _param_count(w: ModelWrapper) -> str:
    n = sum(p.numel() for p in w.model.parameters())
    return f"{n / 1e6:.2f}M"


def _parse_metrics(path: Path):
    if not path.exists():
        return None, None, None
    text = path.read_text(encoding="utf-8")

    def _grab(label: str):
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*([\d.]+)", text)
        return float(m.group(1)) if m else None

    return _grab("Perplexity"), _grab("Corpus BLEU"), _grab("Avg Repetition")


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1a1035 50%, #0f172a 100%);
    color: #e2e8f0;
}

h1, h2, h3, h4, h5, h6 { color: #c7d2fe !important; font-weight: 600 !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30, 41, 59, 0.6);
    padding: 8px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.2s ease;
    border: 1px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #e0e7ff !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
}
div[data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

.stTextInput input, .stTextArea textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 1px #818cf8 !important;
}

.stButton button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}

table {
    background: rgba(30, 41, 59, 0.7) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
th {
    background: rgba(15, 23, 42, 0.8) !important;
    color: #c7d2fe !important;
}
td, th { border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important; }

div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

.block-container { padding-top: 4rem !important; padding-bottom: 2rem !important; }

.stat-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(49, 30, 129, 0.4));
    border: 1px solid rgba(129, 140, 248, 0.25);
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
    margin-bottom: 1rem;
}
.stat-card:hover { transform: translateY(-2px); border-color: rgba(129, 140, 248, 0.5); }
.stat-value { font-size: 2.2rem; font-weight: 700; color: #818cf8; }
.stat-label { font-size: 0.9rem; color: #94a3b8; margin-top: 6px; }
.stat-sub   { font-size: 0.75rem; color: #64748b; margin-top: 6px; }

.gen-output {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 10px;
    padding: 20px 24px;
    font-family: 'Courier New', monospace;
    font-size: 0.92rem;
    color: #e2e8f0;
    line-height: 1.8;
    white-space: pre-wrap;
    margin-top: 1rem;
}

.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #a5b4fc;
    margin: 3px;
}
</style>
"""


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Dáil LLM", page_icon="🏛️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    tab_about, tab_gen, tab_eval, tab_attn = st.tabs([
        "🏠  About",
        "✍️  Text Generation",
        "📊  Evaluation Results",
        "👁️  Attention Visualisation",
    ])

    # -----------------------------------------------------------------------
    # TAB 1 — ABOUT
    # -----------------------------------------------------------------------
    with tab_about:
        st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem;">
  <div style="font-size:3.5rem; margin-bottom:0.5rem;">🏛️</div>
  <h1 style="font-size:2.4rem; margin-bottom:0.3rem;">Dáil LLM</h1>
  <p style="color:#94a3b8; font-size:1.05rem; margin-bottom:1.2rem;">
    Irish Parliamentary Transformer &mdash; trained from scratch on nearly a century of debate
  </p>
  <span class="hero-badge">📅 1919–2013</span>
  <span class="hero-badge">🗣️ 4.4M speeches</span>
  <span class="hero-badge">🔢 3.26M parameters</span>
  <span class="hero-badge">💻 CPU only</span>
  <span class="hero-badge">🔤 No pretrained weights</span>
</div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
I built this to actually understand how transformers work from the inside, not just read about them. So I wrote everything myself in PyTorch: the character-level tokenizer, the multi-head self-attention, the feed-forward blocks, the training loop with checkpointing, and the evaluation pipeline.

The training data is the complete record of the Dáil Éireann, Ireland's lower house of parliament, from January 1919 through to 2013. Nearly a century of political debate, 4.4 million speeches, every elected TD. The model learns to continue those speeches one character at a time with no pretrained weights and no external APIs. Everything runs locally on CPU.
        """)

        st.subheader("🗃️ Dataset")
        col_d1, col_d2 = st.columns([3, 2])
        with col_d1:
            st.markdown(f"""
**{DATASET_NAME}**

The full dataset is 3.44 GB of tab-separated text from Harvard Dataverse, covering 4,443,713 speeches from 1,178 elected TDs spanning January 1919 to 2013.

I extracted about 6.1 MB of English speeches from 1950 onwards. Earlier debates have a much higher proportion of Irish language content, so those were filtered out using a non-ASCII character ratio check per speech. The extracted text was then split 90 / 5 / 5 into train, validation and test sets.

> **Citation:** Proksch S.O. and Slapin J.B. (2010). *Dáil Debates 1919–2013*, Harvard Dataverse.
            """)
        with col_d2:
            st.table({
                "Detail": ["Total speeches", "Unique TDs", "Full date range", "Date range used",
                           "Extracted subset", "Split", "Language filter"],
                "Value":  ["4,443,713", "1,178", "Jan 1919 to 2013", "1950 to 2013",
                           "6.1 MB", "90 / 5 / 5", "English only"],
            })

        st.divider()
        st.subheader("🧠 Model Architecture")

        try:
            wrapper = get_model()
            n_params = _param_count(wrapper)
        except Exception:
            n_params = "model not loaded"

        cfg = [
            ("Type",               "Character-level decoder-only transformer"),
            ("Transformer layers", N_LAYERS),
            ("Attention heads",    N_HEADS),
            ("Embedding dim",      EMBED_DIM),
            ("Head dimension",     EMBED_DIM // N_HEADS),
            ("Context window",     f"{BLOCK_SIZE} characters"),
            ("Feed-forward",       f"4× ({EMBED_DIM} → {EMBED_DIM * 4} → {EMBED_DIM})"),
            ("Activation",         "GELU"),
            ("Dropout",            0.1),
            ("Tokenizer",          "Character-level"),
            ("Positional encoding","Learned embeddings"),
            ("Total parameters",   n_params),
            ("Optimiser",          "AdamW  lr = 3e-4"),
            ("Training steps",     "2,000"),
            ("Batch size",         32),
        ]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            half = len(cfg) // 2
            st.table({"Property": [r[0] for r in cfg[:half]],
                      "Value":    [str(r[1]) for r in cfg[:half]]})
        with col_m2:
            st.table({"Property": [r[0] for r in cfg[half:]],
                      "Value":    [str(r[1]) for r in cfg[half:]]})

    # -----------------------------------------------------------------------
    # TAB 2 — TEXT GENERATION
    # -----------------------------------------------------------------------
    with tab_gen:
        st.title("✍️ Text Generation")
        st.markdown("Type a seed prompt and the model will continue from there, one character at a time.")

        example_prompts = [
            "The Minister for",
            "In this House we",
            "I wish to raise the matter of",
            "On the question of",
        ]

        prompt = st.text_area("Seed prompt", value=example_prompts[0], height=80)

        col_a, col_b = st.columns(2)
        with col_a:
            max_tokens = st.slider("Tokens to generate", 50, 500, 200, step=25)
        with col_b:
            temperature = st.slider(
                "Temperature", 0.5, 1.5, 0.8, step=0.05,
                help="Higher = more creative. Lower = more repetitive.",
            )

        st.caption("Try: " + "  ·  ".join(f'*"{p}"*' for p in example_prompts))

        if st.button("Generate ✨", type="primary"):
            if not prompt.strip():
                st.warning("Add something to the seed prompt first.")
            else:
                try:
                    wrapper = get_model()
                    t0 = time.time()
                    with st.spinner("Running..."):
                        output = wrapper.generate(
                            prompt, max_new_tokens=max_tokens, temperature=temperature
                        )
                    elapsed = time.time() - t0
                    import html as _html
                    safe = _html.escape(output)
                    st.markdown(f'<div class="gen-output">{safe}</div>', unsafe_allow_html=True)
                    st.caption(f"Generated {max_tokens} tokens in {elapsed:.1f}s  ·  temperature {temperature}")
                except FileNotFoundError:
                    st.error("No trained model found. Run `python train_pipeline.py` first.")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    # -----------------------------------------------------------------------
    # TAB 3 — EVALUATION RESULTS
    # -----------------------------------------------------------------------
    with tab_eval:
        st.title("📊 Evaluation Results")

        ppl, bleu, rep = _parse_metrics(EVAL_RESULTS_PATH)

        if any(v is not None for v in (ppl, bleu, rep)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
<div class="stat-card">
  <div class="stat-value">{f"{ppl:.2f}" if ppl is not None else "N/A"}</div>
  <div class="stat-label">📉 Perplexity</div>
  <div class="stat-sub">Lower is better &nbsp;·&nbsp; random baseline ≈ 80</div>
</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
<div class="stat-card">
  <div class="stat-value">{f"{bleu:.4f}" if bleu is not None else "N/A"}</div>
  <div class="stat-label">📝 Corpus BLEU</div>
  <div class="stat-sub">Near zero expected for char-level models</div>
</div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
<div class="stat-card">
  <div class="stat-value">{f"{rep:.4f}" if rep is not None else "N/A"}</div>
  <div class="stat-label">🔁 Repetition Score</div>
  <div class="stat-sub">0.0000 = no repeated 3-gram patterns</div>
</div>""", unsafe_allow_html=True)

            st.divider()

        if EVAL_RESULTS_PATH.exists():
            st.markdown(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        else:
            st.info("No evaluation results yet. Run the evaluation script after training.")

        st.divider()
        st.subheader("📈 Training Curves")
        col1, col2 = st.columns(2)
        with col1:
            p = PLOTS_DIR / "loss.png"
            if p.exists():
                st.image(str(p), caption="Train and Val Loss over 2,000 steps")
            else:
                st.info("loss.png not found — train the model first.")
        with col2:
            p = PLOTS_DIR / "val_perplexity.png"
            if p.exists():
                st.image(str(p), caption="Validation Perplexity over 2,000 steps")
            else:
                st.info("val_perplexity.png not found — train the model first.")

    # -----------------------------------------------------------------------
    # TAB 4 — ATTENTION VISUALISATION
    # -----------------------------------------------------------------------
    with tab_attn:
        st.title("👁️ Attention Visualisation")

        if not _HAS_SEABORN:
            st.error("seaborn is not installed. Run `pip install seaborn` and restart the app.")
            st.stop()

        st.markdown("""
At each transformer layer, every character decides how much it should attend to each character that came before it. Causal masking means it can only look backwards, never ahead. The heatmap below shows those weights as a grid: rows are the query (attending from), columns are the key (attended to), and brighter cells mean stronger attention.

A few patterns worth looking for. Diagonal lines mean the model is mostly attending to the character right before it. Horizontal streaks mean one character is spreading attention broadly across everything before it. Bright column bands are positions that many characters all attend back to, often word boundaries or punctuation marks.
        """)

        attn_prompt = st.text_input("Prompt to visualise", value="The Minister for Finance")

        col1, col2 = st.columns(2)
        with col1:
            sel_layer = st.selectbox("Layer", list(range(N_LAYERS)), index=0)
        with col2:
            sel_head = st.selectbox("Head", list(range(N_HEADS)), index=0)

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            if st.button("Visualise single head", type="primary"):
                if not attn_prompt.strip():
                    st.warning("Enter a prompt above.")
                else:
                    try:
                        wrapper = get_model()
                        fig = visualise_attention(
                            wrapper.model, wrapper.tokenizer,
                            attn_prompt, layer=sel_layer, head=sel_head,
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    except FileNotFoundError:
                        st.error("Model not found — train first.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col_b2:
            if st.button("Show all heads"):
                if not attn_prompt.strip():
                    st.warning("Enter a prompt above.")
                else:
                    try:
                        wrapper = get_model()
                        fig = visualise_all_heads(
                            wrapper.model, wrapper.tokenizer,
                            attn_prompt, layer=sel_layer,
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    except FileNotFoundError:
                        st.error("Model not found — train first.")
                    except Exception as e:
                        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
