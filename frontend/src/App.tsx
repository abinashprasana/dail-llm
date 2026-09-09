import { Route, Routes, useLocation } from "react-router-dom";
import { lazy, Suspense, useEffect } from "react";
import { HomePage } from "./pages/HomePage";

const LabPage = lazy(() => import("./pages/LabPage").then(module => ({ default: module.LabPage })));

const ResearchPage = lazy(() => import("./pages/ResearchPage"));

function ScrollToTop() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) {
      const frame = requestAnimationFrame(() => document.getElementById(decodeURIComponent(hash.slice(1)))?.scrollIntoView({ behavior: "instant" }));
      return () => cancelAnimationFrame(frame);
    }
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [pathname, hash]);
  return null;
}

export default function App() {
  return (
    <>
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/lab" element={<Suspense fallback={<main className="page-width" style={{ paddingTop: 150 }}><p role="status">Loading model lab…</p></main>}><LabPage /></Suspense>} />
        <Route path="/research" element={<Suspense fallback={<main className="page-width" style={{ paddingTop: 150 }}><p role="status">Loading Research…</p></main>}><ResearchPage /></Suspense>} />
        <Route path="*" element={<HomePage />} />
      </Routes>
    </>
  );
}
