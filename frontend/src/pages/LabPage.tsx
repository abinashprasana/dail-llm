import {
  AlertCircle,
  BarChart3,
  Check,
  ChevronRight,
  Clipboard,
  Eye,
  Gauge,
  LoaderCircle,
  Sparkles,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { AttentionCanvas } from "../components/AttentionCanvas";
import { SiteHeader } from "../components/SiteHeader";
import { formatMetric } from "../format";
import type { AttentionResult, EvaluationReport, GenerationResult, HealthStatus, ModelMetadata } from "../types";

type LabTab = "generate" | "evaluation" | "attention";

const tabs: Array<{ id: LabTab; label: string; icon: typeof Sparkles }> = [
  { id: "generate", label: "Generate", icon: Sparkles },
  { id: "evaluation", label: "Evaluation", icon: BarChart3 },
  { id: "attention", label: "Attention", icon: Eye },
];

const promptOptions = ["The Minister for", "In this House", "I wish to raise", "On the matter of"];

const REVEAL_INTERVAL_MS = 12;
const REVEAL_CHARS_PER_TICK = 3;
const COPY_FEEDBACK_MS = 1600;
const COPY_FAILURE_MESSAGE = "The text couldn't be copied. Select it and copy it manually.";
const visuallyHidden = {
  position: "absolute",
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: "hidden",
  clip: "rect(0, 0, 0, 0)",
  whiteSpace: "nowrap",
  border: 0,
} as const;

function PanelError({ message }: { message: string }) {
  return <div className="panel-error" role="alert"><AlertCircle size={18} /><span>{message}</span></div>;
}

function GenerationPanel() {
  const reducedMotion = useReducedMotion();
  const [prompt, setPrompt] = useState(promptOptions[0]);
  const [tokens, setTokens] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [visibleText, setVisibleText] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const copyFeedbackTimer = useRef<number | null>(null);

  useEffect(() => {
    if (!result) {
      setVisibleText("");
      return;
    }

    const announceCompletion = () => {
      const unit = result.generated_characters === 1 ? "character" : "characters";
      setAnnouncement(`Generation complete. ${result.generated_characters} new ${unit}.`);
    };

    if (reducedMotion || result.text.length === 0) {
      setVisibleText(result.text);
      announceCompletion();
      return;
    }

    setVisibleText("");
    let index = 0;
    const timer = window.setInterval(() => {
      index = Math.min(result.text.length, index + REVEAL_CHARS_PER_TICK);
      setVisibleText(result.text.slice(0, index));
      if (index >= result.text.length) {
        window.clearInterval(timer);
        announceCompletion();
      }
    }, REVEAL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [result, reducedMotion]);

  useEffect(() => () => {
    if (copyFeedbackTimer.current !== null) {
      window.clearTimeout(copyFeedbackTimer.current);
      copyFeedbackTimer.current = null;
    }
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (copyFeedbackTimer.current !== null) {
      window.clearTimeout(copyFeedbackTimer.current);
      copyFeedbackTimer.current = null;
    }
    setError("");
    setResult(null);
    setVisibleText("");
    setCopied(false);
    setCopyError("");
    setAnnouncement("");
    if (!prompt.trim()) {
      setError("Enter a prompt before running the model.");
      return;
    }
    setLoading(true);
    setAnnouncement("The model is running.");
    try {
      setResult(await api.generate({ prompt, max_new_tokens: tokens, temperature }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Generation failed.");
    } finally {
      setLoading(false);
    }
  }

  async function copyOutput() {
    if (!result) return;
    if (copyFeedbackTimer.current !== null) {
      window.clearTimeout(copyFeedbackTimer.current);
      copyFeedbackTimer.current = null;
    }
    setError("");
    setCopyError("");
    setAnnouncement("");

    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      setAnnouncement(copied ? "Generated text copied again." : "Generated text copied.");
      copyFeedbackTimer.current = window.setTimeout(() => {
        setCopied(false);
        setAnnouncement((current) => current.startsWith("Generated text copied") ? "" : current);
        copyFeedbackTimer.current = null;
      }, COPY_FEEDBACK_MS);
    } catch {
      setCopied(false);
      setCopyError(COPY_FAILURE_MESSAGE);
      setAnnouncement(COPY_FAILURE_MESSAGE);
    }
  }

  const revealComplete = !result || visibleText.length >= result.text.length;

  return (
    <div className="lab-panel-grid">
      <form className="control-card" onSubmit={submit}>
        <div className="panel-kicker">Generation controls</div>
        <h2>Continue a parliamentary prompt</h2>
        <p>The model predicts one character at a time from the preceding context.</p>

        <label className="field-label" htmlFor="seed-prompt">Seed prompt <span>{prompt.length}/256</span></label>
        <textarea id="seed-prompt" value={prompt} maxLength={256} onChange={(event) => setPrompt(event.target.value)} rows={5} />
        <div className="prompt-chips" aria-label="Suggested prompts">
          {promptOptions.map((option) => (
            <button type="button" key={option} onClick={() => setPrompt(option)}>{option}</button>
          ))}
        </div>

        <div className="range-field">
          <label htmlFor="token-count">New characters <strong>{tokens}</strong></label>
          <input id="token-count" type="range" min="50" max="500" step="25" value={tokens} onChange={(event) => setTokens(Number(event.target.value))} />
          <div><span>50</span><span>500</span></div>
        </div>
        <div className="range-field">
          <label htmlFor="temperature">Temperature <strong>{temperature.toFixed(2)}</strong></label>
          <input id="temperature" type="range" min="0.5" max="1.5" step="0.05" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
          <div><span>Precise</span><span>Variable</span></div>
        </div>
        {error && <PanelError message={error} />}
        <button className="button button-primary run-button" type="submit" disabled={loading}>
          {loading ? <><LoaderCircle className="spin" size={18} /> Model is running</> : <>Generate text <ChevronRight size={18} /></>}
        </button>
      </form>

      <div className="output-card">
        <p
          className="sr-only"
          role="status"
          aria-live="polite"
          aria-atomic="true"
          style={visuallyHidden}
        >
          {announcement}
        </p>
        <div className="output-header">
          <div><span>Model output</span>{result && <small>{result.generated_characters} characters · {(result.elapsed_ms / 1000).toFixed(1)}s</small>}</div>
          <button type="button" onClick={copyOutput} disabled={!result || !revealComplete} aria-label="Copy generated text">
            {copied ? <Check size={17} /> : <Clipboard size={17} />} {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <div className={`output-body ${loading ? "is-loading" : ""}`}>
          {loading && (
            <div className="model-pulse" aria-label="Model calculation in progress">
              <i /><i /><i /><span>Reading the context</span>
            </div>
          )}
          {!loading && !result && <div className="empty-output"><Sparkles size={23} /><p>Your generated passage will appear here.</p></div>}
          {!loading && result && <pre>{visibleText}{!revealComplete && <span className="typing-caret" aria-hidden="true" />}</pre>}
        </div>
        {copyError && <div className="filtered-note">{copyError}</div>}
        {result?.filtered_characters.length ? (
          <div className="filtered-note">Unsupported characters were omitted: {result.filtered_characters.join(" ")}</div>
        ) : null}
      </div>
    </div>
  );
}

function EvaluationPanel({ report, error }: { report: EvaluationReport | null; error: string }) {
  if (error) return <PanelError message={error} />;
  if (!report) return <div className="loading-panel"><LoaderCircle className="spin" /> Loading evaluation record</div>;
  const metrics = [
    ["Cross-entropy", formatMetric(report.metrics.cross_entropy, 4), "Average held-out loss"],
    ["Perplexity", formatMetric(report.metrics.perplexity, 2), "Prediction uncertainty"],
    ["Bits / character", formatMetric(report.metrics.bits_per_character, 3), "Information per prediction"],
    ["Next-character accuracy", formatMetric(report.metrics.next_character_accuracy, 1, { scale: 100, suffix: "%" }), "Exact held-out predictions"],
  ];
  return (
    <div className="evaluation-panel">
      <div className="evaluation-intro">
        <div><div className="panel-kicker">Held-out test record</div><h2>Evidence tied to this checkpoint</h2></div>
        <p>Each measure comes from the test split. Saved samples use a fixed seed and can be reproduced from the same weights.</p>
      </div>
      <div className="lab-metric-grid">
        {metrics.map(([label, value, note]) => <article key={label}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>)}
      </div>
      <div className="research-figures">
        <figure><img src="/research/loss.png" alt="Training and validation loss across 2,000 steps" /><figcaption><strong>Figure 01</strong> Training and validation loss</figcaption></figure>
        <figure><img src="/research/val_perplexity.png" alt="Validation perplexity across 2,000 steps" /><figcaption><strong>Figure 02</strong> Validation perplexity</figcaption></figure>
      </div>
      <div className="sample-list">
        <h3>Deterministic samples</h3>
        {report.samples.map((sample, index) => (
          <details key={sample.prompt} open={index === 0}>
            <summary><span>0{index + 1}</span> {sample.prompt}<ChevronRight size={17} /></summary>
            <blockquote>{sample.text}</blockquote>
          </details>
        ))}
      </div>
    </div>
  );
}

function AttentionPanel({ model }: { model: ModelMetadata | null }) {
  const layers = model?.architecture.n_layers ?? 4;
  const heads = model?.architecture.n_heads ?? 8;
  const [prompt, setPrompt] = useState("The Minister for Finance");
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState<number | "all">(0);
  const [result, setResult] = useState<AttentionResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setResult(null);
    if (!prompt.trim()) {
      setError("Enter a prompt before calculating attention.");
      return;
    }
    setLoading(true);
    try {
      setResult(await api.attention({ prompt, layer, head: head === "all" ? null : head }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Attention could not be calculated.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="attention-panel">
      <div className="evaluation-intro">
        <div><div className="panel-kicker">Causal attention</div><h2>See what each character reads</h2></div>
        <p>Rows are attending characters. Columns are earlier characters available inside the causal mask.</p>
      </div>
      <form className="attention-controls" onSubmit={submit}>
        <label htmlFor="attention-prompt">Prompt<input id="attention-prompt" value={prompt} maxLength={256} onChange={(event) => setPrompt(event.target.value)} /></label>
        <label htmlFor="layer">Layer<select id="layer" value={layer} onChange={(event) => setLayer(Number(event.target.value))}>{Array.from({ length: layers }, (_, index) => <option key={index} value={index}>Layer {index + 1}</option>)}</select></label>
        <label htmlFor="head">Head<select id="head" value={head} onChange={(event) => setHead(event.target.value === "all" ? "all" : Number(event.target.value))}><option value="all">All heads</option>{Array.from({ length: heads }, (_, index) => <option key={index} value={index}>Head {index + 1}</option>)}</select></label>
        <button className="button button-primary" type="submit" disabled={loading}>{loading ? <LoaderCircle className="spin" size={18} /> : <Gauge size={18} />} Calculate</button>
      </form>
      {error && <PanelError message={error} />}
      {loading && <div className="loading-panel"><LoaderCircle className="spin" /> Calculating attention weights</div>}
      {!loading && !result && <div className="attention-empty"><Eye size={25} /><p>Choose a layer and head, then calculate the attention matrix.</p></div>}
      {!loading && result && <AttentionCanvas matrices={result.matrices} labels={result.labels} layer={result.layer} head={result.head} />}
    </div>
  );
}

export function LabPage() {
  const reducedMotion = Boolean(useReducedMotion());
  const [activeTab, setActiveTab] = useState<LabTab>("generate");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationReport | null>(null);
  const [evaluationError, setEvaluationError] = useState("");

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealthError(true));
    api.model().then(setModel).catch(() => undefined);
    api.evaluation().then(setEvaluation).catch((error) => setEvaluationError(error instanceof Error ? error.message : "Evaluation record is unavailable."));
  }, []);

  function moveTab(event: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    setActiveTab(tabs[next].id);
    const buttons = event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>("[role='tab']");
    buttons?.[next]?.focus();
  }

  const healthState = healthError || (health !== null && !health.model_loaded)
    ? "offline"
    : health?.model_loaded
      ? "ready"
      : "connecting";
  const healthLabel = healthState === "offline"
    ? healthError ? "Connection failed" : "Model unavailable"
    : healthState === "ready" ? "Model ready" : "Connecting";
  const healthDetail = healthError
    ? "API unreachable"
    : health
      ? `${health.device} · v${health.version}`
      : "Checking the local service";

  return (
    <div className="site-shell lab-shell">
      <SiteHeader />
      <main id="main-content" className="lab-main">
        <div className="page-width lab-heading">
          <div>
            <div className="eyebrow"><span /> Live model workspace</div>
            <h1>Model lab</h1>
            <p>Generate text, inspect held-out evidence, and read causal attention from the active checkpoint.</p>
          </div>
          <div
            className={`model-status is-${healthState}`}
            role="status"
            aria-live="polite"
            aria-atomic="true"
          >
            <i aria-hidden="true" />
            <div>
              <span>{healthLabel}</span>
              <small>{healthDetail}</small>
            </div>
          </div>
        </div>

        <div className="page-width lab-workspace">
          <div className="lab-tabs" role="tablist" aria-label="Model tools">
            {tabs.map((tab, index) => {
              const Icon = tab.icon;
              return (
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === tab.id}
                  aria-controls={`panel-${tab.id}`}
                  id={`tab-${tab.id}`}
                  tabIndex={activeTab === tab.id ? 0 : -1}
                  key={tab.id}
                  className={activeTab === tab.id ? "is-active" : ""}
                  onClick={() => setActiveTab(tab.id)}
                  onKeyDown={(event) => moveTab(event, index)}
                >
                  <Icon size={18} /> {tab.label}
                </button>
              );
            })}
          </div>

          <AnimatePresence mode="popLayout" initial={false}>
            <motion.section
              key={activeTab}
              id={`panel-${activeTab}`}
              role="tabpanel"
              aria-labelledby={`tab-${activeTab}`}
              initial={{ opacity: 0, y: reducedMotion ? 0 : 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: reducedMotion ? 0 : -4 }}
              transition={{ duration: reducedMotion ? 0.12 : 0.18, ease: [0.22, 1, 0.36, 1] }}
              className="lab-tab-panel"
            >
              {activeTab === "generate" && <GenerationPanel />}
              {activeTab === "evaluation" && <EvaluationPanel report={evaluation} error={evaluationError} />}
              {activeTab === "attention" && <AttentionPanel model={model} />}
            </motion.section>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
