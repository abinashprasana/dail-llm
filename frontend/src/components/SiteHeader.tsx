import { Menu, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Seal } from "./Seal";

export function SiteHeader() {
  const [open, setOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const { pathname } = useLocation();

  useEffect(() => {
    if (!open) return;

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setOpen(false);
      menuButton.current?.focus();
    };

    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [open]);

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header className="site-header">
        <div className="header-inner">
        <Link className="wordmark" to="/" aria-label="Dáil LLM home" onClick={() => setOpen(false)}>
          <Seal size={30} />
          <span>Dáil LLM</span>
        </Link>

        <button
          ref={menuButton}
          className="menu-button"
          type="button"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          aria-controls="primary-navigation"
          onClick={() => setOpen((value) => !value)}
        >
          {open ? <X size={21} /> : <Menu size={21} />}
        </button>

        <nav id="primary-navigation" className={`site-nav ${open ? "is-open" : ""}`} aria-label="Primary navigation">
          <Link to="/ask" aria-current={pathname === "/ask" ? "page" : undefined} onClick={() => setOpen(false)}>Ask the debates</Link>
          <Link to="/lab" aria-current={pathname === "/lab" ? "page" : undefined} onClick={() => setOpen(false)}>Model Lab</Link>
          <Link to="/research" aria-current={pathname === "/research" ? "page" : undefined} onClick={() => setOpen(false)}>Research</Link>
          <a href="/#data" onClick={() => setOpen(false)}>Data</a>
          <Link className="nav-home" to="/" aria-current={pathname === "/" ? "page" : undefined} onClick={() => setOpen(false)}>Overview</Link>
        </nav>
        </div>
      </header>
    </>
  );
}
