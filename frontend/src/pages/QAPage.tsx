import { FormEvent, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import { SiteHeader } from "../components/SiteHeader";
import type { QACapabilities, QAResult } from "../types";
import "../qa.css";

export default function QAPage() {
  const [capabilities, setCapabilities] = useState<QACapabilities | null>(null);
  const [question, setQuestion] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [result, setResult] = useState<QAResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.qaCapabilities().then(setCapabilities).catch(() => setError("Debate search is unavailable."));
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await api.qaAsk({ question: question.trim(), start_date: start || null, end_date: end || null }));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "The request could not be completed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="site-shell">
      <SiteHeader />
      <main id="main-content" className="qa-page page-width">
        <p className="qa-kicker">Official Report · Dáil Éireann</p>
        <h1>Ask the debates</h1>
        <p className="qa-intro">Explore what speakers said, with links to the original debate. Answers describe the record; they do not express political opinions.</p>
        <div className="qa-coverage" role="status">
          {capabilities?.coverage?.days
            ? `Indexed ${capabilities.coverage.passages.toLocaleString()} passages from ${capabilities.coverage.first_date} to ${capabilities.coverage.last_date}.`
            : "The current-debates index is not available on this deployment."}
        </div>
        <form className="qa-form" onSubmit={submit}>
          <label htmlFor="qa-question">Your question</label>
          <textarea id="qa-question" value={question} onChange={event => setQuestion(event.target.value)} minLength={5} maxLength={500} required placeholder="What was said about housing in 2025?" />
          <div className="qa-filters">
            <label>From <input type="date" min="2014-01-01" value={start} onChange={event => setStart(event.target.value)} /></label>
            <label>To <input type="date" min="2014-01-01" value={end} onChange={event => setEnd(event.target.value)} /></label>
            <button className="button button-primary" type="submit" disabled={loading || !capabilities?.available}>{loading ? "Searching…" : "Search debates"}</button>
          </div>
        </form>
        {error && <p className="qa-error" role="alert">{error}</p>}
        {result && <section className="qa-results" aria-live="polite">
          <h2>{result.status === "answered" ? "Answer from the record" : result.status === "sources_only" ? "Relevant passages" : "Insufficient evidence"}</h2>
          {result.answer ? <p className="qa-answer">{result.answer}</p> : <p>No supporting passage was found in the indexed dates. Try a narrower topic or a different date range.</p>}
          {result.sources.length > 0 && <div className="qa-source-list">
            {result.sources.map(source => <article key={source.passage_id} className="qa-source">
              <div className="qa-source-meta"><span>{source.date}</span><span>{source.speaker}</span></div>
              <h3>{source.title}</h3>
              <p>{source.text.length > 650 ? `${source.text.slice(0, 650)}…` : source.text}</p>
              <a href={source.source_url} target="_blank" rel="noreferrer">Read the Official Report ↗</a>
              <small>{source.passage_id}</small>
            </article>)}
          </div>}
          <p className="qa-note">A listed passage may mention the topic without supporting every possible claim. Check the full source before relying on it.</p>
        </section>}
      </main>
    </div>
  );
}
