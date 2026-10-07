import { ArrowRight, BookOpen, CalendarDays, Search } from "lucide-react";
import { FormEvent, Fragment, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import { SiteHeader } from "../components/SiteHeader";
import { SiteFooter } from "../components/SiteFooter";
import type { QACapabilities, QAResult } from "../types";
import "../qa.css";

function formatDate(value: string) {
  const parsed = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? value : new Intl.DateTimeFormat("en-IE", {
    day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
  }).format(parsed);
}

function CitedAnswer({ result }: { result: QAResult }) {
  const sourceNumbers = new Map(result.sources.map((source, index) => [source.passage_id, index + 1]));
  return <div className="qa-answer">{result.answer?.split(/(\[[^\]\n]+\])/g).map((part, index) => {
    const id = part.startsWith("[") && part.endsWith("]") ? part.slice(1, -1) : "";
    const number = sourceNumbers.get(id);
    return <Fragment key={index}>{number && result.citation_ids.includes(id)
      ? <a className="qa-citation" href={`#qa-source-${number}`} aria-label={`Jump to source ${number}`}>[{number}]</a>
      : part}</Fragment>;
  })}</div>;
}

export default function QAPage() {
  const [capabilities, setCapabilities] = useState<QACapabilities | null>(null);
  const [capabilityError, setCapabilityError] = useState(false);
  const [question, setQuestion] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [result, setResult] = useState<QAResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState<string[]>([]);

  useEffect(() => {
    api.qaCapabilities().then(setCapabilities).catch(() => setCapabilityError(true));
  }, []);

  const dateError = start && end && start > end ? "The start date must be before the end date." : "";
  const coverage = capabilities?.coverage;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (dateError) return;
    setLoading(true);
    setError("");
    setResult(null);
    setExpanded([]);
    try {
      setResult(await api.qaAsk({ question: question.trim(), start_date: start || null, end_date: end || null }));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "The request could not be completed.");
    } finally {
      setLoading(false);
    }
  }

  function toggleSource(id: string) {
    setExpanded(current => current.includes(id) ? current.filter(item => item !== id) : [...current, id]);
  }

  return (
    <div className="site-shell">
      <SiteHeader />
      <main id="main-content" className="qa-page page-width">
        <header className="qa-heading">
          <div>
            <p className="qa-kicker">Official Report · Dáil Éireann</p>
            <h1>Ask the debates</h1>
            <p className="qa-intro">Search what speakers said and follow each result to the original record. Responses describe the debates; they do not express political opinions.</p>
          </div>
          <BookOpen className="qa-heading-mark" size={68} strokeWidth={1} aria-hidden="true" />
        </header>
        <div className="qa-coverage" role="status">
          <CalendarDays size={18} aria-hidden="true" />
          <span>{capabilityError || (capabilities && !capabilities.available)
            ? "Debate search is unavailable on this deployment."
            : coverage?.days
              ? `Indexed ${coverage.passages.toLocaleString("en-IE")} passages from ${formatDate(coverage.first_date!)} to ${formatDate(coverage.last_date!)}.`
              : "Checking the connected debate index…"}</span>
        </div>
        <form className="qa-form" onSubmit={submit}>
          <div className="qa-form-heading"><div><span className="qa-kicker">Search the record</span><h2>What would you like to find?</h2></div><Search size={25} aria-hidden="true" /></div>
          <label htmlFor="qa-question">Your question</label>
          <textarea id="qa-question" value={question} onChange={event => setQuestion(event.target.value)} minLength={5} maxLength={500} required placeholder="What was said about housing in 2025?" />
          <div className="qa-filters">
            <label>From <input type="date" min="2014-01-01" max={end || undefined} value={start} onChange={event => setStart(event.target.value)} aria-invalid={Boolean(dateError)} /></label>
            <label>To <input type="date" min={start || "2014-01-01"} value={end} onChange={event => setEnd(event.target.value)} aria-invalid={Boolean(dateError)} /></label>
            <button className="button button-primary" type="submit" disabled={loading || !capabilities?.available || Boolean(dateError)}>{loading ? "Searching…" : <>Search debates <ArrowRight size={16} /></>}</button>
          </div>
          {dateError && <p className="qa-error" role="alert">{dateError}</p>}
          <p className="qa-form-note">Dates narrow the search. The available record is shown above.</p>
        </form>
        {error && <p className="qa-error qa-request-error" role="alert">{error}</p>}
        {result && <section className="qa-results" aria-live="polite" aria-labelledby="qa-results-heading">
          <div className="qa-result-heading"><span className="qa-kicker">Search result</span><h2 id="qa-results-heading">{result.status === "answered" ? "Answer from the record" : result.status === "sources_only" ? "Relevant passages" : "Insufficient evidence"}</h2></div>
          <div className="qa-result-grid">
            <div className="qa-answer-panel">
              {result.status === "sources_only" && <p className="qa-state-label">Showing cited excerpts</p>}
              {result.status === "answered" && <p className="qa-state-label">Generated answer · check the cited passages</p>}
              {result.answer ? <CitedAnswer result={result} /> : <p className="qa-empty">No supporting passage was found in the indexed dates. Try a narrower topic or another date range.</p>}
              {result.status === "sources_only" && <p className="qa-answer-note">Answer generation is unavailable. These excerpts come directly from retrieved passages.</p>}
            </div>
            {result.sources.length > 0 && <div className="qa-source-list" aria-label="Source passages">
              {result.sources.map((source, index) => {
                const open = expanded.includes(source.passage_id);
                const long = source.text.length > 650;
                return <article id={`qa-source-${index + 1}`} key={source.passage_id} className="qa-source">
                  <div className="qa-source-meta"><span className="qa-source-number">{String(index + 1).padStart(2, "0")}</span><span>{formatDate(source.date)}</span><span>{source.speaker}</span></div>
                  <h3>{source.title}</h3>
                  <p>{long && !open ? `${source.text.slice(0, 650).trimEnd()}…` : source.text}</p>
                  {long && <button className="qa-expand" type="button" aria-expanded={open} onClick={() => toggleSource(source.passage_id)}>{open ? "Show less" : "Read more of this passage"}</button>}
                  <a href={source.source_url} target="_blank" rel="noreferrer">Read the Official Report ↗</a>
                </article>;
              })}
            </div>}
          </div>
          {result.sources.length > 0 && <p className="qa-note">A passage can mention the topic without supporting every claim. Check the full Official Report before relying on it.</p>}
        </section>}
      </main>
      <SiteFooter />
    </div>
  );
}
