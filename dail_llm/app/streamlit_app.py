"""
Dáil LLM — Irish Parliamentary Transformer — Streamlit Dashboard

Run from the project root:
    streamlit run dail_llm/app/streamlit_app.py
"""
import sys
import io
import time
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CKPT_PATH, PLOTS_DIR, EVAL_RESULTS_PATH,
    N_LAYERS, N_HEADS, EMBED_DIM, BLOCK_SIZE, DATASET_NAME,
)
from dail_llm.inference import ModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def get_model_wrapper():
    return ModelWrapper()


def _param_count(wrapper: ModelWrapper) -> str:
    n = sum(p.numel() for p in wrapper.model.parameters())
    return f"{n / 1e6:.2f}M"


def _attn_fig(wrapper: ModelWrapper, prompt: str, layer: int, head: int):
    """Return a matplotlib Figure with the attention heatmap."""
    try:
        import seaborn as sns
    except ImportError:
        return None

    model = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    ids = ids[: model.block_size].unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, all_att = model(ids, return_attention_weights=True)

    att = all_att[layer][0, head].cpu().numpy()
    chars = [tokenizer.itos[i.item()] for i in ids[0]]
    labels = [repr(c)[1:-1] if c in (" ", "\n", "\t") else c for c in chars]

    size = max(6, len(chars) // 2)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(att, xticklabels=labels, yticklabels=labels,
                ax=ax, cmap="Blues", vmin=0.0, linewidths=0.3, linecolor="white")
    ax.set_title(f"Layer {layer}, Head {head} — Attention Weights")
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    plt.tight_layout()
    return fig


def _all_heads_fig(wrapper: ModelWrapper, prompt: str, layer: int):
    try:
        import seaborn as sns
    except ImportError:
        return None

    model = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    ids = ids[: model.block_size].unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, all_att = model(ids, return_attention_weights=True)

    att_layer = all_att[layer][0].cpu().numpy()
    n_heads = att_layer.shape[0]
    chars = [tokenizer.itos[i.item()] for i in ids[0]]
    labels = [repr(c)[1:-1] if c in (" ", "\n", "\t") else c for c in chars]

    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols
    cell = max(3, len(chars) // 3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell))
    if n_heads == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    flat = [ax for row in axes for ax in row]

    for h, ax in enumerate(flat):
        if h < n_heads:
            sns.heatmap(att_layer[h], xticklabels=labels, yticklabels=labels,
                        ax=ax, cmap="Blues", cbar=False, linewidths=0.2, linecolor="white")
            ax.set_title(f"Head {h}", fontsize=9)
            ax.tick_params(axis="both", labelsize=6)
        else:
            ax.set_visible(False)

    fig.suptitle(f"Layer {layer} — All Heads", fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Dáil LLM",
        page_icon="🏛️",
        layout="wide",
    )

    st.markdown("""
        <style>
        /* Import a nice font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* App Background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #e2e8f0;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #c7d2fe !important;
            font-weight: 600 !important;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(30, 41, 59, 0.6);
            padding: 8px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
            border-radius: 8px;
            padding: 10px 20px;
            transition: all 0.3s ease;
            border: 1px solid transparent !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(99, 102, 241, 0.2) !important;
            color: #e0e7ff !important;
            border: 1px solid rgba(129, 140, 248, 0.4) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        }

        /* Hide the default Streamlit tab highlight and border lines */
        div[data-baseweb="tab-highlight"] {
            display: none !important;
            background-color: transparent !important;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
            background-color: transparent !important;
        }

        /* Cards and expanders */
        .stExpander, div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            backdrop-filter: blur(12px);
            padding: 15px;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea {
            background-color: rgba(15, 23, 42, 0.6) !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 8px !important;
        }

        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #818cf8 !important;
            box-shadow: 0 0 0 1px #818cf8 !important;
        }

        /* Buttons */
        .stButton button {
            background: linear-gradient(to right, #4f46e5, #6366f1) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39) !important;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5) !important;
        }

        /* Tables */
        table {
            background: rgba(30, 41, 59, 0.7) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #e2e8f0 !important;
        }
        th {
            background: rgba(15, 23, 42, 0.8) !important;
            color: #c7d2fe !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        td {
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        /* General layout spacing */
        .block-container {
            padding-top: 5rem !important;
            padding-bottom: 2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    tab_about, tab_gen, tab_eval, tab_attn = st.tabs([
        "About",
        "Text Generation",
        "Evaluation Results",
        "Attention Visualisation",
    ])

    # -----------------------------------------------------------------------
    # TAB 1 — ABOUT
    # -----------------------------------------------------------------------
    with tab_about:
        st.title("Dáil LLM Irish Parliamentary Transformer")

        st.markdown("""
Hey everyone I am Krishna and this is a small experiment I built to understand how transformers work from the inside out. I trained a character level language model on almost a century of Irish parliamentary debates and wired up proper evaluation and attention visualisation so I could see what the model was actually learning rather than just hoping it worked. 

The goal was never to compete with large models like GPT. The goal was to understand how attention selects which characters to focus on, what perplexity actually measures, what repetition looks like, and why it happens. This project is my answer to those questions, written in PyTorch and about twelve hundred lines of code.
        """)

        st.subheader("Dataset")
        st.markdown(f"""
**{DATASET_NAME}**

The Dáil Éireann or Irish parliament debates dataset covers every recorded speech from the first session in January 1919 through to 2013. That is 4.4 million speeches in total from over 1,100 elected representatives or TDs.

The full dataset is 3.44 GB of tab separated text from Harvard Dataverse. For this project I extracted approximately **6 MB** of English language speeches from 1950 onwards, filtering out Irish language content by checking the non ASCII character ratio per speech.

**Citation:** Dáil Debates 1919 through 2013, Harvard Dataverse.
        """)

        st.subheader("Model Architecture")

        try:
            wrapper = get_model_wrapper()
            n_params = _param_count(wrapper)
        except Exception:
            n_params = "Model not loaded"

        cfg_rows = [
            ("Transformer layers", N_LAYERS),
            ("Attention heads", N_HEADS),
            ("Embedding dimension", EMBED_DIM),
            ("Context window in chars", BLOCK_SIZE),
            ("Dropout", 0.1),
            ("Tokenizer", "Character level"),
            ("Total parameters", n_params),
        ]
        col1, col2 = st.columns(2)
        with col1:
            st.table({"Property": [r[0] for r in cfg_rows],
                      "Value":    [str(r[1]) for r in cfg_rows]})

        st.subheader("Why I built this")
        st.markdown("""
I wanted a project that forced me to implement every component myself rather than calling a fit function. Building the tokenizer, the attention mechanism, the training loop, and the evaluation pipeline from scratch taught me more about transformers than any tutorial did.

The Dáil debates dataset was a deliberate choice. It is domain specific, historically interesting, and has real language quirks like Irish names, legal phrasing, and bilingual sections that make the task non trivial.
        """)

    # -----------------------------------------------------------------------
    # TAB 2 — TEXT GENERATION
    # -----------------------------------------------------------------------
    with tab_gen:
        st.title("Text Generation")
        st.markdown("Run the trained model forward from a seed prompt.")

        example_prompts = [
            "The Minister for",
            "In this House we",
            "I wish to raise the matter of",
            "On the question of",
        ]
        prompt = st.text_area(
            "Enter a seed prompt",
            value=example_prompts[0],
            height=80,
            help="The model will continue from wherever you stop.",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            max_tokens = st.slider("Max new tokens", 50, 500, 200, step=25)
        with col_b:
            temperature = st.slider("Temperature", 0.5, 1.5, 0.8, step=0.05,
                                    help="Higher = more creative / unpredictable.")

        st.caption("Examples: " + " · ".join(f'"{p}"' for p in example_prompts))

        if st.button("Generate", type="primary"):
            if not prompt.strip():
                st.warning("Please enter a prompt.")
            else:
                try:
                    wrapper = get_model_wrapper()
                    t0 = time.time()
                    with st.spinner("Generating…"):
                        output = wrapper.generate(prompt, max_new_tokens=max_tokens,
                                                  temperature=temperature)
                    elapsed = time.time() - t0
                    st.success(output)
                    st.caption(f"Generated in {elapsed:.1f} s")
                except FileNotFoundError:
                    st.error("No checkpoint found please train the model first.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # -----------------------------------------------------------------------
    # TAB 3 — EVALUATION RESULTS
    # -----------------------------------------------------------------------
    with tab_eval:
        st.title("Evaluation Results")

        if EVAL_RESULTS_PATH.exists():
            md_text = EVAL_RESULTS_PATH.read_text(encoding="utf-8")

            # Parse metric cards from the markdown table
            lines = md_text.splitlines()
            ppl_val = bleu_val = rep_val = None
            for line in lines:
                if "Perplexity" in line and "|" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        try:
                            ppl_val = float(parts[1])
                        except ValueError:
                            pass
                if "Corpus BLEU" in line and "|" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        try:
                            bleu_val = float(parts[1])
                        except ValueError:
                            pass
                if "Avg Repetition" in line and "|" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        try:
                            rep_val = float(parts[1])
                        except ValueError:
                            pass

            if ppl_val or bleu_val or rep_val:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Perplexity", f"{ppl_val:.1f}" if ppl_val else "Not available",
                              help="Lower is better. exp avg cross entropy loss.")
                with m2:
                    st.metric("Corpus BLEU", f"{bleu_val:.4f}" if bleu_val else "Not available",
                              help="Higher is better. Zero means no overlap, one means perfect.")
                with m3:
                    st.metric("Avg Repetition", f"{rep_val:.4f}" if rep_val else "Not available",
                              help="Lower is better. Fraction of repeated sequences.")

            st.divider()
            st.markdown(md_text)
        else:
            st.info(
                "No evaluation results yet.  "
                "Run python evaluate script after training."
            )

        st.divider()
        st.subheader("Training curves")

        loss_png = PLOTS_DIR / "loss.png"
        ppl_png = PLOTS_DIR / "val_perplexity.png"

        col1, col2 = st.columns(2)
        with col1:
            if loss_png.exists():
                st.image(str(loss_png), caption="Train and Val Loss")
            else:
                st.info("loss.png not found train first.")
        with col2:
            if ppl_png.exists():
                st.image(str(ppl_png), caption="Validation Perplexity")
            else:
                st.info("val perplexity.png not found train first.")

    # -----------------------------------------------------------------------
    # TAB 4 — ATTENTION VISUALISATION
    # -----------------------------------------------------------------------
    with tab_attn:
        st.title("Attention Visualisation")

        st.markdown("""
**What are attention weights?**

At each layer, the transformer decides how much each character should pay attention to every other character that came before it. Causal masking means it can only look backwards. The heatmap below shows those weights. A bright cell at a specific row and column means that character is strongly attending to another character.

Here are some patterns to look for. Diagonal lines indicate attending to the immediately previous character or local context. Horizontal streaks mean one character is attending broadly to all previous characters. Vertical streaks mean many characters are all attending to the same key position like a word boundary.
        """)

        attn_prompt = st.text_input(
            "Enter a prompt to visualise attention",
            value="The Minister for Finance",
        )

        col1, col2 = st.columns(2)
        with col1:
            sel_layer = st.selectbox("Layer", list(range(N_LAYERS)), index=0)
        with col2:
            sel_head = st.selectbox("Head", list(range(N_HEADS)), index=0)

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("Visualise single head", type="primary"):
                if not attn_prompt.strip():
                    st.warning("Please enter a prompt.")
                else:
                    try:
                        wrapper = get_model_wrapper()
                        fig = _attn_fig(wrapper, attn_prompt, sel_layer, sel_head)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.error("seaborn is required: pip install seaborn")
                    except FileNotFoundError:
                        st.error("Model not found train first.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with col_btn2:
            if st.button("Show all heads"):
                if not attn_prompt.strip():
                    st.warning("Please enter a prompt.")
                else:
                    try:
                        wrapper = get_model_wrapper()
                        fig = _all_heads_fig(wrapper, attn_prompt, sel_layer)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.error("seaborn is required: pip install seaborn")
                    except FileNotFoundError:
                        st.error("Model not found train first.")
                    except Exception as e:
                        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
