import { Link } from "react-router-dom";

export function SiteFooter() {
  return <footer className="site-footer">
    <div className="page-width footer-grid">
      <div>
        <span className="footer-name">Dáil LLM</span>
        <p>Explore the parliamentary record, the model, and the research.</p>
        <a href="https://github.com/abinashprasana/dail-llm" target="_blank" rel="noreferrer">Source on GitHub ↗</a>
      </div>
      <div className="footer-links" aria-label="Explore Dáil LLM">
        <Link to="/ask">Ask the debates</Link>
        <Link to="/lab">Model Lab</Link>
        <Link to="/research">Research</Link>
      </div>
      <div className="footer-credit">
        <p>Historical archive: Alexander Herzog and Slava J. Mikhaylov (2017), Harvard Dataverse.</p>
        <a href="https://doi.org/10.7910/DVN/6MZN76" target="_blank" rel="noreferrer">Dataset DOI ↗</a>
      </div>
    </div>
  </footer>;
}
