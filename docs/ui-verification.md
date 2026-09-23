# Research interface verification

The interface integrates the completed CPU pilot while keeping the served checkpoint and its published evaluation artifacts separate. The new Research page includes recorded source inspection, optional live inspection, historical comparisons, and methods. The homepage retains its chamber, archive treatment, and model lab, with a scroll-led explanation of speech memory.

## Checks completed

| Check | Result |
|---|---|
| Python suite | 55 passed, including model, data, publication, API, and compatibility checks |
| Frontend unit suite | 41 passed |
| Browser suite | 73 passed; 31 viewport-specific cases skipped |
| TypeScript, ESLint, Ruff | Passed |
| Production build and bundle budgets | Passed |
| Public artifact audit | All 24 recorded inspections and 741 source exclusions matched fresh computations |
| Production HTTP checks | Passed with live inspection enabled and with recorded examples only |

The public artifact audit verifies file hashes, probabilities, source spans, and neighbours after removal. It also checks that the original run's file hashes and modification times remain unchanged. The underlying pilot separately passed 50 scientific artifact checks, described in [the pilot results](pilot-results.md).

The Python suite covers deterministic resume, character positions, unknown handling, speech boundaries, partition ownership, training-only memory, source exclusion, historical cohorts, and legacy generation and attention. Publication tests include incompatible artifacts and private runtime bundle validation. API tests cover input bounds, release mismatches, unavailable research assets, and shared inference capacity.

Browser checks cover four research views at desktop, laptop, tablet, and mobile sizes. They exercise source removal and restoration, deep links, keyboard navigation, enlarged text, reduced motion, busy responses, Unicode input limits, and stale requests. Automated accessibility checks passed on those views. Existing chamber fallbacks, manual trace controls, native scrolling, and archive behaviour also passed.

Desktop and mobile snapshots were reviewed. The changed hero copy, removed ornamental labels, and adjusted mobile gutters account for the three updated existing baselines. Research views, the memory workspace, and three scroll stages have separate baselines. Focused workspace captures hide the fixed header during capture so it cannot obscure the component; full-page checks retain the header.

## Build sizes

| Asset | Raw bytes | Gzip bytes |
|---|---:|---:|
| Initial JavaScript, including static imports | 406,597 | 131,668 |
| Application CSS | 39,991 | 9,584 |
| Optional chamber JavaScript | 877,380 | 236,386 |
| Research JavaScript | 28,437 | 9,270 |
| Research CSS | 10,859 | 2,718 |
| Memory explanation JavaScript | 3,867 | 1,817 |
| Memory explanation CSS | 1,871 | 688 |

The public summary is 33,430 bytes; the largest recorded example is 149,655 bytes. The build verifies each example's hash and enforces payload limits. Research code and the memory explanation load separately from the initial page. Application CSS is close to its 40,000-byte budget, so further additions will need care.

## Limits and release status

Docker is unavailable on the local machine. Both image configurations have build and HTTP smoke checks in CI, but remote CI and container execution were not run here. Local HTTP checks verified production routes, legacy plots, generation, public release identity, and the live research endpoint. These checks do not replace a container run on the deployment host.

One existing Starlette/httpx deprecation warning remains. Vite reports the large optional chamber chunk; its separate size budget passes. Automated accessibility checks and snapshots do not establish accessibility for every assistive technology or browser. The browser suite uses Chromium, including mobile viewport emulation.

Execution dates and date-stamped run names are absent from reader-facing research copy and downloads. Historical periods and source speech dates remain visible because they are part of the evidence. Internal execution records retain their provenance.

No model was retrained or promoted during this interface update. The full study and deployment were not launched. Pilot findings remain exploratory: memory-policy uncertainty intervals cross zero, the five-gram baseline performs better, and historical matching yields one pair.

See [the interface guide](research-ui.md) for reproduction and deployment commands.
