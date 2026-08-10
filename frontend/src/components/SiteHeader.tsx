import { Menu, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";

const links = [
  { label: "Model", href: "/#model" },
  { label: "Data", href: "/#data" },
  { label: "Evidence", href: "/#evidence" },
];

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
          <span className="wordmark-seal" aria-hidden="true"><i /><i /><i /></span>
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
          {links.map((link) => (
            <a key={link.label} href={link.href} onClick={() => setOpen(false)}>{link.label}</a>
          ))}
          <Link
            className={`nav-lab ${pathname === "/lab" ? "is-active" : ""}`}
            to="/lab"
            onClick={() => setOpen(false)}
          >
            Open model lab
          </Link>
        </nav>
        </div>
      </header>
    </>
  );
}
