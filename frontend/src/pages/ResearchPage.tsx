import { ArrowRight, ArrowUpRight, Download, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { SiteHeader } from "../components/SiteHeader";
import {
  characterLabel,
  fetchJSON,
  fetchRecording,
  number,
  policyNames,
  recordedResult,
  researchRoot,
  sourceSegments,
  visibleDistributions,
} from "../research";
import type {
  Capabilities,
  Inspection,
  Metrics,
  Neighbour,
  Policy,
  Recording,
  Summary,
} from "../research";
import "../research.css";

const views = {
  overview: "Overview",
  memory: "Speech memory",
  history: "Historical evaluation",
  methods: "Methods",
};
const systems: Record<string, string> = {
  ngram: "Witten–Bell five-gram",
  none: "Unconditioned decoder",
  party: "Party-conditioned",
  role: "Role-conditioned",
};
const policies = Object.keys(policyNames) as Policy[];

function ErrorNotice({ children }: { children: string }) {
  return (
    <p className="research-notice" role="alert">
      {children}
    </p>
  );
}
function SourceSpan({ source }: { source: Neighbour }) {
  const { before, target, after } = sourceSegments(source);
  return (
    <blockquote>
      {before}
      <mark>{target}</mark>
      {after}
    </blockquote>
  );
}
function MetricTable({
  data,
}: {
  data: { label: string; metrics: Metrics | undefined }[];
}) {
  return (
    <div
      className="table-scroll"
      tabIndex={0}
      role="region"
      aria-label="Model measurements"
    >
      <table className="research-table">
        <caption>Supported-target bits per character. Lower is better.</caption>
        <thead>
          <tr>
            <th scope="col">System</th>
            <th scope="col">BPC</th>
            <th scope="col">Coverage</th>
            <th scope="col">Unknown</th>
          </tr>
        </thead>
        <tbody>
          {data.map(({ label, metrics }) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{number(metrics?.supported_target_bpc)}</td>
              <td>
                {metrics?.coverage == null
                  ? "Unavailable"
                  : `${(metrics.coverage * 100).toFixed(5)}%`}
              </td>
              <td>{metrics?.unknown ?? "Unavailable"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Overview({ summary }: { summary: Summary }) {
  return (
    <>
      <div className="research-intro">
        <h2>
          Two questions.
          <br />
          <em>One traceable experiment.</em>
        </h2>
        <p>
          Can a small memory of speeches help a character model? And how well
          does prediction hold up across a change of government? These studies
          keep the model, data splits, and comparisons explicit.
        </p>
      </div>
      <div className="study-index">
        <Link to="?view=memory" className="study-row">
          <span className="study-number">01</span>
          <div>
            <h3>What does speech memory contribute?</h3>
            <p>
              Compare three ways of selecting the same number of memory entries.
              Inspect a prediction, remove a source, and follow the change.
            </p>
            <span className="text-link">
              Inspect speech memory <ArrowRight size={16} />
            </span>
          </div>
          <span className="study-detail">
            {summary.memory.uniform.entries.toLocaleString()} entries
            <br />3 selection policies
          </span>
        </Link>
        <Link to="?view=history" className="study-row">
          <span className="study-number">02</span>
          <div>
            <h3>What changes across historical periods?</h3>
            <p>
              Compare unconditioned and metadata-conditioned decoders on
              speeches before and after the 2011 government transition.
            </p>
            <span className="text-link">
              Read the historical comparison <ArrowRight size={16} />
            </span>
          </div>
          <span className="study-detail">
            4 fixed periods
            <br />1 transition
          </span>
        </Link>
      </div>
      <div className="research-finding">
        <span className="research-kicker">What the pilot tells us</span>
        <h3>
          The measurement works.
          <br />
          The improvement is still an open question.
        </h3>
        <p>
          The five-gram baseline outperformed the transformers at this training
          budget. Both memory-selection intervals include zero, and the
          historical matching found only {summary.cohort.pairs} pair. These
          results support further testing, not a claim that either study has
          established an improvement.
        </p>
        <Link to="?view=methods" className="text-link">
          Read the settings and limitations <ArrowRight size={16} />
        </Link>
      </div>
      <MetricTable
        data={[
          {
            label: systems.ngram,
            metrics: summary.history.ngram.splits.earlier.aggregate,
          },
          {
            label: systems.none,
            metrics: summary.history.none.splits.earlier.aggregate,
          },
          ...policies.map((p) => ({
            label: policyNames[p],
            metrics: summary.memory[p].metrics,
          })),
        ]}
      />
      <p className="research-note">
        Earlier test only. These research scores use different data preparation
        and accounting from the published model lab; they are not directly
        comparable.
      </p>
    </>
  );
}

function ProbabilityView({ result }: { result: Inspection }) {
  const rows = visibleDistributions(result.distributions);
  const removed = result.excluded_speech !== null;
  return (
    <section
      className="probability-panel"
      aria-label="Next-character probabilities"
    >
      <div className="research-panel-heading">
        <h3>What comes next?</h3>
        <span>Probability, without rescaling</span>
      </div>
      <div className="probability-legend">
        <span className="base-key">Model only</span>
        <span className="memory-key">With memory</span>
        {removed && <span className="removed-key">After removal</span>}
      </div>
      <div className="probability-rows">
        {rows.map((row) => (
          <div className="probability-row" key={row.character}>
            <span className={`character-name${characterLabel(row.character) !== row.character ? " is-label" : ""}`}>
              {characterLabel(row.character)}
            </span>
            <div className="probability-tracks" aria-hidden="true">
              <i className="base-bar" style={{ width: `${row.base * 100}%` }} />
              <i
                className="memory-bar"
                style={{ width: `${row.before * 100}%` }}
              />
              {removed && (
                <i
                  className="removed-bar"
                  style={{ width: `${row.after * 100}%` }}
                />
              )}
            </div>
            <span className="probability-values">
              <span>
                {number(row.base * 100, 2)}% / {number(row.before * 100, 2)}%
                {removed && ` / ${number(row.after * 100, 2)}%`}
              </span>
              {removed && (
                <small>
                  {row.delta >= 0 ? "+" : ""}
                  {number(row.delta * 100, 2)} pp
                </small>
              )}
            </span>
          </div>
        ))}
      </div>
      <p className="research-note">
        Other characters:{" "}
        {number((1 - rows.reduce((s, d) => s + d.base, 0)) * 100, 2)}% model
        only; {number((1 - rows.reduce((s, d) => s + d.before, 0)) * 100, 2)}%
        with memory
        {removed &&
          `; ${number((1 - rows.reduce((s, d) => s + d.after, 0)) * 100, 2)}% after removal`}
        . Rows include the top eight characters from each distribution.
      </p>
    </section>
  );
}

function MemoryView({ summary }: { summary: Summary }) {
  const [policy, setPolicy] = useState<Policy>("uniform");
  const [exampleId, setExampleId] = useState(summary.examples[0].id);
  const [recording, setRecording] = useState<Recording | null>(null);
  const [excluded, setExcluded] = useState<string | null>(null);
  const [showAllSources, setShowAllSources] = useState(false);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [live, setLive] = useState(false);
  const [prefix, setPrefix] = useState("The Minister for");
  const [liveResult, setLiveResult] = useState<Inspection | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);
  const example = summary.examples.find((e) => e.id === exampleId)!;
  useEffect(() => {
    const controller = new AbortController();
    fetchJSON<Capabilities>("/api/v1/research/capabilities", controller.signal)
      .then(setCapabilities)
      .catch(() => {});
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setRecording(null);
    setExcluded(null);
    setShowAllSources(false);
    setError("");
    fetchRecording(example, policy, summary, controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setRecording(value);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [example, policy, summary]);
  useEffect(() => () => request.current?.abort(), []);
  const cancelLive = () => {
    request.current?.abort();
    setBusy(false);
    setLiveResult(null);
    setExcluded(null);
    setError("");
  };
  const inspect = async (speech: string | null = null) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const timeout = window.setTimeout(
      () => controller.abort("timeout"),
      30_000,
    );
    setBusy(true);
    setError("");
    try {
      const value = await fetchJSON<Inspection>(
        "/api/v1/research/inspect",
        controller.signal,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            release_id: summary.release_id,
            policy,
            prefix,
            excluded_speech: speech,
          }),
        },
      );
      if (controller.signal.aborted) return;
      if (
        value.release_id !== summary.release_id ||
        value.policy !== policy ||
        value.prefix !== prefix
      )
        throw new Error("The response does not match this research request.");
      setLiveResult(value);
      setExcluded(speech);
    } catch (e) {
      if (controller.signal.reason === "timeout")
        setError(
          "Inspection took too long. Your prefix is kept; try again shortly.",
        );
      else if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      window.clearTimeout(timeout);
      if (request.current === controller) setBusy(false);
    }
  };
  const matchingRecording =
    recording?.original.policy === policy &&
    recording?.original.prefix === example.prefix;
  const result = live
    ? liveResult
    : recording && matchingRecording
      ? recordedResult(recording, excluded)
      : null;
  const eligible =
    capabilities?.live && capabilities.release_id === summary.release_id;
  const members = result
    ? [
        ...new Map(
          result.neighbours_before.map((n) => [n.speech_id, n]),
        ).values(),
      ]
    : [];
  const shownMembers = showAllSources ? members : members.slice(0, 6);
  return (
    <>
      <div className="research-intro">
        <h2>
          A prediction with
          <br />
          <em>its sources in view.</em>
        </h2>
        <p>
          Memory adds evidence from nearby character contexts. Choose a
          selection policy, then remove one contributing speech to see how the
          next-character probabilities change.
        </p>
      </div>
      <div className="memory-workspace">
        <div className="memory-controls">
          <div className="research-panel-heading">
            <h3>{live ? "Live inspection" : "Recorded example"}</h3>
            {eligible && (
              <button
                type="button"
                className="text-link"
                onClick={() => {
                  cancelLive();
                  setLive(!live);
                }}
              >
                {live ? "Use recorded examples" : "Try your own prefix"}{" "}
                <ArrowUpRight size={15} />
              </button>
            )}
          </div>
          {!live ? (
            <label>
              Example
              <select
                value={exampleId}
                onChange={(e) => {
                  setExcluded(null);
                  setExampleId(e.target.value);
                }}
              >
                {summary.examples.map((e, i) => (
                  <option key={e.id} value={e.id}>
                    {i + 1}. {e.kind}
                  </option>
                ))}
              </select>
              <blockquote className="prefix-text">
                {example.prefix}
                <span aria-hidden="true" className="prefix-cursor" />
              </blockquote>
            </label>
          ) : (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void inspect();
              }}
            >
              <label htmlFor="research-prefix">Your prefix</label>
              <textarea
                id="research-prefix"
                value={prefix}
                onChange={(e) => {
                  cancelLive();
                  setPrefix(e.target.value);
                }}
                aria-describedby="prefix-count"
                rows={3}
              />
              <div className="prefix-form-end">
                <small id="prefix-count">
                  {Array.from(prefix).length} / 256 characters
                </small>
                <button
                  className="button button-primary"
                  disabled={busy || !prefix || Array.from(prefix).length > 256}
                >
                  Inspect prefix <ArrowRight size={15} />
                </button>
              </div>
            </form>
          )}
          <label>
            Memory selection
            <select
              value={policy}
              onChange={(e) => {
                cancelLive();
                setPolicy(e.target.value as Policy);
              }}
            >
              {policies.map((p) => (
                <option key={p} value={p}>
                  {policyNames[p]}
                </option>
              ))}
            </select>
          </label>
          <p className="research-note">
            {summary.memory[policy].entries.toLocaleString()} training entries.{" "}
            {summary.memory[policy].parameters.k} neighbours, interpolation
            weight {summary.memory[policy].parameters.weight}, temperature{" "}
            {summary.memory[policy].parameters.temperature}. Settings were
            chosen on validation data.
          </p>
          {!eligible && (
            <p className="research-note">
              Recorded examples include recomputed source removals. Custom
              prefixes become available when a compatible research backend is
              connected.
            </p>
          )}
          {error && <ErrorNotice>{error}</ErrorNotice>}
          <p className="research-status" role="status">
            {busy
              ? "Inspecting the prefix…"
              : result
                ? `${result.removed_entries} entries removed. ${result.remaining_entries.toLocaleString()} remain. ${result.unknown_characters} unknown input characters.`
                : live
                  ? "Enter a prefix to inspect."
                  : error
                    ? "Example unavailable."
                    : "Loading the recorded example…"}
          </p>
        </div>
        {result && <ProbabilityView result={result} />}
      </div>
      {result && (
        <section className="source-section">
          <div className="research-panel-heading">
            <div>
              <h3>Contributing speeches</h3>
              <p>
                Remove one source at a time. Every entry from that speech is
                excluded before searching again.
              </p>
            </div>
            <button
              className="research-reset"
              disabled={!excluded || busy}
              onClick={() => (live ? void inspect(null) : setExcluded(null))}
            >
              <RotateCcw size={15} /> Restore source
            </button>
          </div>
          <div className="source-list">
            {shownMembers.map((source) => (
              <article
                className={`source-card ${excluded === source.speech_id ? "is-excluded" : ""}`}
                key={source.speech_id}
              >
                <div className="source-byline">
                  <span>{source.member || "Recorded member unavailable"}</span>
                  <time dateTime={source.date}>{source.date}</time>
                </div>
                <p className="source-title">{source.title}</p>
                <SourceSpan source={source} />
                <div className="source-card-end">
                  <small>
                    Speech {source.speech_id} · offset {source.offset}
                  </small>
                  <button
                    aria-pressed={excluded === source.speech_id}
                    disabled={busy}
                    onClick={() =>
                      live
                        ? void inspect(
                            excluded === source.speech_id
                              ? null
                              : source.speech_id,
                          )
                        : setExcluded(
                            excluded === source.speech_id
                              ? null
                              : source.speech_id,
                          )
                    }
                  >
                    {excluded === source.speech_id
                      ? "Restore"
                      : "Exclude speech"}
                  </button>
                </div>
              </article>
            ))}
          </div>
          {members.length > 6 && (
            <button
              className="research-reset source-expand"
              aria-expanded={showAllSources}
              onClick={() => setShowAllSources((value) => !value)}
            >
              {showAllSources
                ? "Show fewer sources"
                : `Show all ${members.length} contributing speeches`}
            </button>
          )}
          {excluded && (
            <details className="research-details">
              <summary>Sources after the new search</summary>
              <ul>
                {[
                  ...new Map(
                    result.neighbours_after.map((n) => [n.speech_id, n]),
                  ).values(),
                ].map((n) => (
                  <li key={n.speech_id}>
                    {n.member} — {n.date}, speech {n.speech_id}
                  </li>
                ))}
              </ul>
              {!result.neighbours_after.length && (
                <p>
                  No eligible memory remains. The distribution falls back to the
                  model.
                </p>
              )}
            </details>
          )}
          <p className="research-note">
            Spans are short excerpts from{" "}
            <a href={summary.source}>the source corpus</a>. Member labels are
            recorded metadata. This measures an effect through retrieval, not
            training-data attribution or factual verification. Unknown input
            characters retain their positions and map to an explicit unknown
            token.
          </p>
        </section>
      )}
      <MemoryEvidence summary={summary} />
    </>
  );
}

function MemoryEvidence({ summary }: { summary: Summary }) {
  return (
    <section className="research-evidence">
      <h3>Selection results, beyond the average</h3>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Memory selection evidence"
      >
        <table className="research-table">
          <caption>Earlier-test measurements at equal memory size</caption>
          <thead>
            <tr>
              <th>Policy</th>
              <th>BPC</th>
              <th>Difference vs uniform</th>
              <th>95% interval</th>
              <th>Rare-word BPC</th>
              <th>Retrieval ms / target</th>
              <th>Generation chars / s</th>
              <th>Copied eight-grams</th>
            </tr>
          </thead>
          <tbody>
            {policies.map((p) => {
              const m = summary.memory[p];
              return (
                <tr key={p}>
                  <th scope="row">{policyNames[p]}</th>
                  <td>{number(m.metrics.supported_target_bpc)}</td>
                  <td>
                    {m.comparison
                      ? number(m.comparison.difference_bpc)
                      : "Reference"}
                  </td>
                  <td>
                    {m.comparison?.interval
                      ?.map((n) => number(n))
                      .join(" to ") ?? "—"}
                  </td>
                  <td>{number(m.slices.word_rare?.bpc)}</td>
                  <td>{number(m.retrieval.ms_per_query, 3)}</td>
                  <td>{number(m.tokens_per_second, 1)}</td>
                  <td>
                    {m.copying == null
                      ? "Unavailable"
                      : `${number(m.copying * 100, 1)}%`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="research-note">
        Both selection intervals include zero. Rare-word results do not improve
        over uniform memory. No recorded full member-name spans matched in this
        earlier test. All systems had zero exact rare-word span accuracy. Slice
        labels are heuristic; zero measured copying does not establish readable
        or original prose. Timing is from a shared CPU and is not a hardware
        benchmark.
      </p>
    </section>
  );
}

function HistoryView({ summary }: { summary: Summary }) {
  const [cohort, setCohort] = useState<
    "aggregate" | "continuing_speakers" | "matched"
  >("aggregate");
  const [split, setSplit] = useState("earlier");
  const counts =
    cohort === "aggregate"
      ? summary.selection[split].speeches
      : cohort === "matched"
        ? summary.cohort.pairs
        : summary.cohort.continuing_speeches[split];
  return (
    <>
      <div className="research-intro">
        <h2>
          Language across
          <br />
          <em>a change of government.</em>
        </h2>
        <p>
          A case study around the 2011 transition. The comparisons account for
          continuing speakers and similar passages, while keeping retrieval
          switched off.
        </p>
      </div>
      <ol className="history-timeline">
        {Object.entries(summary.periods).map(([key, dates]) => (
          <li key={key}>
            <span>
              {
                {
                  train: "Training",
                  validation: "Validation",
                  earlier: "Earlier test",
                  later: "Later test",
                }[key]
              }
            </span>
            <strong>
              {dates[0].slice(0, 7)}
              <br />
              {dates[1].slice(0, 7)}
            </strong>
            <small>{summary.selection[key].speeches} speeches</small>
          </li>
        ))}
      </ol>
      <p className="timeline-note">
        Government transition:{" "}
        <a href="https://www.oireachtas.ie/en/debates/debate/dail/2011-03-09/4/">
          9 March 2011
        </a>
        . January–March 2011 is outside the selected test periods. The timeline
        shows partitions, not a continuous time scale.
      </p>
      <div className="history-controls">
        <label>
          Comparison
          <select
            value={cohort}
            onChange={(e) => setCohort(e.target.value as typeof cohort)}
          >
            <option value="aggregate">All speeches</option>
            <option value="continuing_speakers">Continuing speakers</option>
            <option value="matched">Matched passages</option>
          </select>
        </label>
        <label>
          Test period
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="earlier">July–December 2010</option>
            <option value="later">April–September 2011</option>
          </select>
        </label>
      </div>
      <p className="research-status" role="status">
        {counts} {cohort === "matched" ? "matched passage" : "speeches"} in this
        view. {summary.cohort.continuing_speakers} speakers appear in both test
        periods.
      </p>
      <MetricTable
        data={Object.entries(systems).map(([key, label]) => ({
          label,
          metrics: summary.history[key].splits[split][cohort],
        }))}
      />
      <div className="research-notice">
        <strong>Only {summary.cohort.pairs} matched pair.</strong>
        <p>
          {summary.cohort.unmatched_earlier} earlier and{" "}
          {summary.cohort.unmatched_later} later passages were unmatched. This
          is too small a cohort to support a role-change conclusion.
        </p>
      </div>
      <div className="method-columns">
        <section>
          <h3>What was controlled?</h3>
          <p>
            The same backbone, training exposure, and held-out periods were used
            for the three decoders. Party and role embeddings add parameters;
            retrieval is disabled.
          </p>
          <dl>
            {["none", "party", "role"].map((mode) => (
              <div key={mode}>
                <dt>{systems[mode]}</dt>
                <dd>
                  {summary.history[mode].parameters.toLocaleString()} parameters
                  (
                  {(
                    summary.history[mode].parameters -
                    summary.history.none.parameters
                  ).toLocaleString()}{" "}
                  added)
                </dd>
              </div>
            ))}
          </dl>
        </section>
        <section>
          <h3>Where the evidence stops</h3>
          <p>
            Passages were matched within speakers using training-fitted TF-IDF,
            without replacement, and with no more than a 25% length difference.
          </p>
          <p>
            Role mappings cover recorded Fianna Fáil, Fine Gael, and Labour
            affiliations. Other affiliations are unmapped. Some archive labels
            are outdated; dated membership verification is still needed before
            drawing political conclusions.
          </p>
        </section>
      </div>
      <details className="research-details">
        <summary>
          Input perturbations: recorded names and recurring phrases
        </summary>
        <p>
          These masks test sensitivity to input text. They do not simulate
          political behaviour. Each masked result is compared with a control
          scoring the same targets.
        </p>
        <div className="table-scroll">
          <table className="research-table">
            <thead>
              <tr>
                <th>Model / mask</th>
                <th>Masked characters</th>
                <th>Control BPC</th>
                <th>Masked-input BPC</th>
              </tr>
            </thead>
            <tbody>
              {["none", "party", "role"].flatMap((mode) =>
                Object.entries(
                  summary.history[mode].splits[split].perturbations,
                ).map(([key, value]) => (
                  <tr key={`${mode}-${key}`}>
                    <th scope="row">
                      {systems[mode]} / {key}
                    </th>
                    <td>{value.masked_characters}</td>
                    <td>
                      {number(value.same_target_control.supported_target_bpc)}
                    </td>
                    <td>{number(value.masked_input.supported_target_bpc)}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

function MethodsView({ summary }: { summary: Summary }) {
  return (
    <>
      <div className="research-intro">
        <h2>
          Every result
          <br />
          <em>has a record.</em>
        </h2>
        <p>
          Configuration, corpus selection, checkpoints, and memory are bound
          together by hashes. Incompatible artifacts are rejected before a
          result reaches this page.
        </p>
      </div>
      <div className="method-columns">
        <section>
          <h3>Training and evaluation</h3>
          <p>
            {summary.steps} steps per decoder, batch size {summary.batch_size},
            seed {summary.seed}. Four layers, eight heads, 256-dimensional
            representations, and a 256-character context. The full three-seed
            study has not been run.
          </p>
          <p>
            Validation selects checkpoints and retrieval settings. Complete
            debate groups stay within one period. Context resets at speech
            boundaries and each eligible target is scored once.
          </p>
          <p>
            Supported-target BPC excludes unknown original characters. Coverage
            and unknown counts are reported alongside it. Mapped-token loss
            measures the tokenizer output; it is not exact likelihood for unseen
            characters.
          </p>
        </section>
        <section>
          <h3>Source preparation</h3>
          <p>
            The archive is read as ten-column tab-separated text with quotation
            processing disabled. Unicode and source identifiers are preserved.
            Language is unclassified parliamentary text.
          </p>
          <p>
            Exact duplicates are removed. Speeches of at least 50 words are also
            checked for five-gram Jaccard similarity of at least 0.9. Earlier
            occurrences are retained; future speeches are never moved into
            training.
          </p>
          <p>
            {summary.counts.duplicate_exclusions} duplicate exclusions.{" "}
            {summary.counts.malformed ?? 0} malformed rows recorded. Whole-group
            character caps favour debates that fit the remaining budget.
          </p>
        </section>
      </div>
      <details className="research-details">
        <summary>Memory selection and measurement</summary>
        <p>
          Uniform positions, positions balanced across speeches, and
          speech-balanced positions with each normalized trailing 64-character
          context capped at four entries. All three use{" "}
          {summary.memory.uniform.entries.toLocaleString()} entries and exact
          squared-Euclidean search.
        </p>
        <p>
          Neighbour count, interpolation weight, and temperature are selected on
          validation data. Paired 95% intervals resample debate groups. The
          pilot uses one seed, so it cannot estimate variation across training
          seeds.
        </p>
      </details>
      <details className="research-details">
        <summary>Validation loss and artifact verification</summary>
        <table className="research-table">
          <caption>
            Unconditioned decoder: deterministic validation checkpoints
          </caption>
          <thead>
            <tr>
              <th>Step</th>
              <th>Training loss</th>
              <th>Validation loss</th>
            </tr>
          </thead>
          <tbody>
            {summary.training_history.map((row) => (
              <tr key={row.step}>
                <td>{row.step}</td>
                <td>{number(row.train_loss)}</td>
                <td>{number(row.validation_loss)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          {summary.verification.checks} artifact checks{" "}
          {summary.verification.status}. Checks cover partition ownership,
          target accounting, checkpoint selection, stored representations, and
          exact retrieval. They do not verify political membership or rate
          generated prose.
        </p>
        <p>
          Recorded execution: {number(summary.execution.elapsed_seconds, 1)}{" "}
          seconds; peak process memory:{" "}
          {number(summary.execution.peak_memory_bytes / 1024 ** 3, 2)} GiB.
          Human review remains pending.
        </p>
      </details>
      <details className="research-details">
        <summary>Input fingerprints</summary>
        <dl className="fingerprints">
          {Object.entries({
            "Source archive": summary.source_sha256,
            "Prepared corpus": summary.provenance.binding.corpus,
            Checkpoint: summary.provenance.checkpoint_sha256,
            Tokenizer: summary.provenance.tokenizer_sha256,
            "Research report": summary.provenance.report_sha256,
            Verification: summary.verification.sha256,
          }).map(([label, hash]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>
                <code>{hash}</code>
              </dd>
            </div>
          ))}
        </dl>
      </details>
      <a
        className="button button-quiet research-download"
        href={`${researchRoot}/summary.json`}
        download="dail-research-results.json"
      >
        <Download size={16} /> Download results and provenance
      </a>
      <section className="research-references">
        <h3>Methods and sources</h3>
        <ul>
          {summary.references.map(([title, url]) => (
            <li key={url}>
              <a href={url} target="_blank" rel="noreferrer">
                {title} <ArrowUpRight size={14} />
              </a>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}

export default function ResearchPage() {
  const [search, setSearch] = useSearchParams();
  const current = search.get("view") ?? "overview";
  const view = Object.hasOwn(views, current)
    ? (current as keyof typeof views)
    : "overview";
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetchJSON<Summary>(`${researchRoot}/summary.json`, controller.signal)
      .then((value) => {
        if (
          value.schema_version !== 1 ||
          !value.examples?.length ||
          value.verification.status !== "passed"
        )
          throw new Error(
            "This research release is incomplete or incompatible.",
          );
        if (!controller.signal.aborted) setSummary(value);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [retry]);
  return (
    <div className="site-shell">
      <SiteHeader />
      <main id="main-content" className="research-main page-width">
        <header className="research-heading">
          <div>
            <span className="research-kicker">Dáil LLM</span>
            <h1>Research</h1>
            <p>
              Speech memory, historical evaluation, and the evidence behind each
              comparison.
            </p>
          </div>
          <div className="research-run-label">
            <strong>CPU pilot</strong>
            <span>Exploratory results</span>
            {summary && (
              <small>
                {summary.steps} steps per model · seed {summary.seed}
              </small>
            )}
          </div>
        </header>
        <div
          className="research-tabs"
          role="tablist"
          aria-label="Research views"
        >
          {Object.entries(views).map(([id, label], index) => (
            <button
              key={id}
              role="tab"
              id={`tab-${id}`}
              aria-selected={view === id}
              tabIndex={view === id ? 0 : -1}
              aria-controls="research-view"
              onClick={() => setSearch(id === "overview" ? {} : { view: id })}
              onKeyDown={(e) => {
                const keys = Object.keys(views);
                const next =
                  e.key === "ArrowRight"
                    ? (index + 1) % keys.length
                    : e.key === "ArrowLeft"
                      ? (index + keys.length - 1) % keys.length
                      : e.key === "Home"
                        ? 0
                        : e.key === "End"
                          ? keys.length - 1
                          : -1;
                if (next >= 0) {
                  e.preventDefault();
                  setSearch({ view: keys[next] });
                  document.getElementById(`tab-${keys[next]}`)?.focus();
                }
              }}
            >
              {label}
            </button>
          ))}
        </div>
        <div
          id="research-view"
          role="tabpanel"
          aria-labelledby={`tab-${view}`}
          tabIndex={0}
          className="research-view"
        >
          {error ? (
            <>
              <ErrorNotice>{error}</ErrorNotice>
              <button
                className="button button-quiet"
                onClick={() => setRetry((n) => n + 1)}
              >
                Try again
              </button>
            </>
          ) : !summary ? (
            <p role="status">Loading the research record…</p>
          ) : view === "memory" ? (
            <MemoryView summary={summary} />
          ) : view === "history" ? (
            <HistoryView summary={summary} />
          ) : view === "methods" ? (
            <MethodsView summary={summary} />
          ) : (
            <Overview summary={summary} />
          )}
        </div>
        <footer className="research-footer">
          <p>
            Research findings and the published model lab use separate
            checkpoints.
          </p>
          <Link to="/lab" className="text-link">
            Open model lab <ArrowRight size={16} />
          </Link>
        </footer>
      </main>
    </div>
  );
}
