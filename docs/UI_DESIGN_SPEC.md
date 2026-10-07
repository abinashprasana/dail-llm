# Dáil LLM interface specification

## Design direction

The site uses a cinematic editorial style drawn from the parliamentary record. Deep green surfaces, brass accents, Newsreader headings, and Manrope interface text carry across all four routes. The chamber is an illustration of the character model's prediction path. It is not a depiction of a real debate or a source for answers.

The home page introduces three routes in this order: Ask the debates, Model Lab, and Research. Each route has a different evidence boundary. Ask links to Official Report passages from the index connected to the deployment. Model Lab runs the 1950 character model. Research shows a separate pilot on selected 2008–2011 debates.

The [Figma composition boards](https://www.figma.com/design/JJLUiML9v4pLBfOzKfe7Ro) show editable desktop and mobile layouts for all four routes, plus shared colour foundations and a primary button component. The running application remains the reference for exact spacing, responsive behaviour, and the chamber illustration.

## Foundations

| Role | Value | Use |
| --- | --- | --- |
| Background | `#07110e` | Main canvas |
| Raised surface | `#0a1914` | Forms and panels |
| Deep green | `#123a2d` | Section depth |
| Brass | `#c9a55c` | Actions, links, focus, selected states |
| Parchment | `#e9e2d0` | Main text |
| Sage | `#afc0b5` | Supporting text |

The page width is at most 1180 px. Reading text is usually limited to about 65 characters per line. Buttons are pill shaped; panels use modest rounded corners. Dividers carry hierarchy before elevation or shadow. Text and form-control contrast must meet WCAG AA.

## Page composition

- **Home, desktop:** two-column hero with concise introduction and chamber; four model facts below it. Three editorial rows provide the main paths. Source provenance, model explanation, held-out measurements, and research links follow.
- **Home, mobile:** introduction and actions precede the chamber. The same three paths stack below it, with full-width touch targets.
- **Ask, desktop:** introduction, deployment coverage, and question form lead into a results area with answer at left and numbered sources at right. The answer links only to matching returned source IDs.
- **Ask, mobile:** form controls and results stack in reading order. Sources follow the answer. Long excerpts can expand without losing the Official Report link.
- **Lab:** the controls and generated text share a workspace on desktop and stack on narrow screens. Generation, evaluation, and attention use one tab system. The output is explicitly labelled as generated model text.
- **Research:** the pilot context and methods link sit close to the introduction. Tabs preserve the existing experiments. Tables use aligned numerals and remain readable at narrow widths.

## States and motion

Search distinguishes loading, unavailable index, valid search, request error, cited excerpts without a generated answer, generated answer, and insufficient evidence. Coverage comes from the connected API response. The interface never substitutes the larger local corpus count for a public sample.

Short interface transitions use 160–240 ms easing. The chamber prediction trace plays once, supports replay, and stops when out of view or when the tab is hidden. Reduced-motion and WebGL fallback use the static chamber. Search results appear immediately, without staged animation.

## Review checklist

Review desktop, tablet, and phone widths for all four routes. Check 200% zoom, keyboard focus order, mobile navigation, date errors, source expansion, long Unicode passages, reduced motion, and WebGL failure. Run lint, unit tests, production build with bundle limits, and browser accessibility and screenshot checks. Update visual baselines only after inspecting the new images.

## October refinement review

A browser review at 1440 px and 390 px identified competing navigation emphasis, broken inline provenance text, low-contrast chamber labels, and an extra grid row separating the closing calls to action. The refinement makes Overview a quiet navigation item, restores normal inline source text, increases annotation contrast and trace touch targets, and groups the Lab links. Compact layouts keep the home actions together where they fit and stack the Lab status beneath its introduction. The question composer uses tighter heading and panel spacing.

No service availability or research claims changed. The local preview correctly reports unavailable API services when the backend is absent.
