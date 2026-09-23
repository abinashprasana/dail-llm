import { ArrowRight, BookOpen, Cpu, Database, Fingerprint, FlaskConical } from "lucide-react";
import { motion, useInView, useReducedMotion } from "motion/react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { ArchitectureFlow } from "../components/ArchitectureFlow";
import { ArchiveStamp } from "../components/ArchiveStamp";
import { HeroScene } from "../components/HeroScene";
import { MetricCard } from "../components/MetricCard";
import { SiteHeader } from "../components/SiteHeader";
import { formatCompactNumber, formatMetric } from "../format";
import type { EvaluationReport, ModelMetadata } from "../types";
import published from "../published-evaluation.json";

const MemorySequence = lazy(() => import("../components/MemorySequence").then(module => ({ default: module.MemorySequence })));

function DeferredMemorySequence() {
  const ref = useRef<HTMLDivElement>(null);
  const visible = useInView(ref, { once: true, margin: "1000px 0px" });
  return <div ref={ref} className="memory-section-slot">{visible && <Suspense fallback={null}><MemorySequence /></Suspense>}</div>;
}

const fallbackArchitecture = {
  block_size: 256,
  embed_dim: 256,
  n_layers: 4,
  n_heads: 8,
  parameters: 3_271_168,
};

export function HomePage() {
  const reducedMotion = Boolean(useReducedMotion());
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationReport | null>(null);
  const [modelError, setModelError] = useState(false);
  const [evaluationError, setEvaluationError] = useState(false);

  useEffect(() => {
    api.model().then(setModel).catch(() => setModelError(true));
    api.evaluation().then(setEvaluation).catch(() => { setEvaluation(published); setEvaluationError(true); });
  }, []);

  const architecture = model?.architecture ?? fallbackArchitecture;
  const corpus = model?.dataset?.corpus;
  const date = (value: string) => new Intl.DateTimeFormat("en-IE", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${value}T00:00:00Z`));
  const dateRange = corpus?.selected_date_min && corpus?.selected_date_max
    ? `${date(corpus.selected_date_min)}–${date(corpus.selected_date_max)}`
    : "15 Feb 1950–25 Apr 1950";
  const corpusSize = corpus?.clean_bytes
    ? `${(corpus.clean_bytes / 1024 / 1024).toFixed(1)} MB`
    : "6.0 MB";
  const speechCount = (corpus?.accepted_speeches ?? 9_080).toLocaleString("en-IE");
  const sample = evaluation?.samples?.[0]?.text ?? published.samples[0].text;

  const stats = useMemo(() => [
    { value: formatCompactNumber(architecture.parameters), label: "parameters" },
    { value: String(architecture.block_size), label: "character context" },
    { value: String(architecture.n_layers), label: "decoder layers" },
    { value: String(architecture.n_heads), label: "attention heads" },
  ], [architecture]);

  return (
    <div className="site-shell">
      <SiteHeader />
      <main id="main-content">
        <section className="hero-section">
          <div className="hero-grid page-width">
            <motion.div
              className="hero-copy"
              initial={{ opacity: 0, y: reducedMotion ? 0 : 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={reducedMotion ? { duration: 0.18 } : undefined}
            >
              <h1>Parliamentary debate, <em>modeled character by character</em></h1>
              <p className="hero-lede">
                A {formatCompactNumber(architecture.parameters)}-parameter transformer trained on Dáil Éireann debates.
                Generate text, inspect attention, and explore studies of speech memory and historical change.
              </p>
              <div className="hero-actions">
                <Link className="button button-primary" to="/lab">
                  Open model lab <ArrowRight size={17} />
                </Link>
                <a className="button button-quiet" href="#model">Read the model notes</a>
              </div>
              {modelError && evaluationError && (
                <p className="service-note">Live model and evaluation data are unavailable. The figures below are the last published record.</p>
              )}
              {modelError && !evaluationError && (
                <p className="service-note">Live model status is unavailable. Evaluation figures below are current.</p>
              )}
              {evaluationError && !modelError && (
                <p className="service-note">Live evaluation is unavailable. Evidence below uses the saved published record.</p>
              )}
              <p className="hero-provenance">
                <a href="#evidence">Verified checkpoint</a>
                <a href="#data">Dáil Éireann debates, {dateRange}</a>
              </p>
            </motion.div>
            <HeroScene />
          </div>
          <div className="hero-stats page-width" aria-label="Model summary">
            {stats.map((stat) => (
              <div className="hero-stat" key={stat.label}><strong>{stat.value}</strong><span>{stat.label}</span></div>
            ))}
          </div>
        </section>

        <section className="position-statement page-width" aria-label="Research position">
          <div className="statement-mark">D</div>
          <p>
            I built every part of this model myself to see how language models work. The most useful finding so far: a simple five-gram model still beats it at this scale.
          </p>
        </section>

        <section className="content-section model-section" id="model">
          <div className="page-width split-heading">
            <div>
              <div className="eyebrow"><span /> Inside the model</div>
              <h2>A compact transformer, shown stage by stage</h2>
            </div>
            <p>
              Every prediction passes through the same four-stage path. The model reads at most 256 characters,
              applies causal attention, and estimates what comes next.
            </p>
          </div>
          <div className="page-width"><ArchitectureFlow /></div>
        </section>

        <section className="content-section data-section" id="data">
          <div className="page-width data-layout">
            <motion.div
              className="archive-card-reveal"
              initial={{ opacity: 0, x: -24 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.35 }}
            >
              <div className="archive-card">
                <div className="archive-topline"><BookOpen size={16} /> Dataset provenance</div>
                <div className="archive-title">Dáil Éireann<br />Parliamentary Debates</div>
                <dl>
                  <div><dt>Source archive</dt><dd>1919–2013</dd></div>
                  <div><dt>Selected material</dt><dd>{dateRange}</dd></div>
                  <div><dt>Accepted speeches</dt><dd>{speechCount}</dd></div>
                  <div><dt>Training corpus</dt><dd>{corpusSize}</dd></div>
                  <div><dt>Unit of language</dt><dd>Character</dd></div>
                </dl>
                <ArchiveStamp reducedMotion={reducedMotion} />
              </div>
            </motion.div>
            <div className="data-copy">
              <div className="eyebrow"><span /> The source material</div>
              <h2>Parliamentary language carries its own rhythm</h2>
              <p>
                The current checkpoint uses a {corpusSize} filtered extraction from debates dated {dateRange}.
                Speeches under 50 characters and speeches above a 40% non-ASCII threshold were excluded.
              </p>
              <p>
                The published corpus is divided into training, validation, and test material. The separate research
                pipeline retains Unicode and uses whole debate groups from 2008–2011.
              </p>
              <a className="text-link" href="https://doi.org/10.7910/DVN/6MZN76" target="_blank" rel="noreferrer">
                View the dataset citation <ArrowRight size={15} />
              </a>
              <Link className="text-link archive-research-link" to="/research?view=methods">Read the research preparation method <ArrowRight size={15} /></Link>
            </div>
          </div>
        </section>

        <DeferredMemorySequence />

        <section className="content-section evidence-section" id="evidence">
          <div className="page-width">
            <div className="split-heading evidence-heading">
              <div>
                <div className="eyebrow"><span /> Published checkpoint</div>
                <h2>Measured on held-out parliamentary text</h2>
              </div>
              <p>
                These measurements belong to the checkpoint served in the model lab. Research results use separate checkpoints and evaluation conventions.
              </p>
            </div>
            {evaluationError && <details className="service-note"><summary>Saved published evaluation</summary><p style={{ overflowWrap: "anywhere" }}>Source SHA-256: {published.source_sha256}</p></details>}
            <div className="metric-grid">
              <MetricCard index={0} label="Perplexity" value={formatMetric(evaluation?.metrics.perplexity, 2)} note="Average uncertainty on held-out characters" />
              <MetricCard index={1} label="Bits per character" value={formatMetric(evaluation?.metrics.bits_per_character, 3)} note="Information required for each prediction" />
              <MetricCard index={2} label="Next-character accuracy" value={formatMetric(evaluation?.metrics.next_character_accuracy, 1, { scale: 100, suffix: "%" })} note="Exact predictions across the test split" />
              <MetricCard index={3} label="Checkpoint" value={evaluation?.checkpoint.name ?? "model_best.pt"} note="The weights used by this public instance" />
            </div>

            <div className="sample-ledger">
              <div className="sample-meta">
                <span>Generated record</span>
                <strong>Prompt: “The Minister for”</strong>
                <small>Temperature 0.8 · 200 new characters</small>
              </div>
              <blockquote>{sample}</blockquote>
              <Link className="sample-cta" to="/lab">Run a prompt <ArrowRight size={16} /></Link>
            </div>
          </div>
        </section>

        <section className="lab-invitation">
          <div className="page-width lab-invitation-inner">
            <div>
              <div className="eyebrow light"><span /> Model lab</div>
              <h2>Inspect the model while it works</h2>
            </div>
            <div className="lab-features">
              <div><Cpu size={19} /><span>Generate locally</span></div>
              <div><Fingerprint size={19} /><span>Read attention</span></div>
              <div><FlaskConical size={19} /><span>Review evidence</span></div>
              <div><Database size={19} /><span>Trace the dataset</span></div>
            </div>
            <Link className="button button-parchment" to="/lab">Enter the model lab <ArrowRight size={17} /></Link>
            <Link className="text-link invitation-research" to="/research">Explore the research <ArrowRight size={16} /></Link>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="page-width footer-grid">
          <div><span className="footer-name">Dáil LLM</span><p>Irish parliamentary language, modeled one character at a time.</p></div>
          <p>Dataset: Alexander Herzog and Slava J. Mikhaylov (2017), Harvard Dataverse.</p>
          <a href="https://doi.org/10.7910/DVN/6MZN76" target="_blank" rel="noreferrer">DOI 10.7910/DVN/6MZN76</a>
        </div>
      </footer>
    </div>
  );
}
