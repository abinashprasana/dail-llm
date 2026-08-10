import type { TraceMotion, TraceStage } from "./heroTrace";

interface SceneFallbackProps {
  visible?: boolean;
  traceStage: TraceStage;
  traceMotion: TraceMotion;
  traceRunId: number;
  running: boolean;
}

const seatTiles = [
  { x: 184, y: 265, rotate: -28, glyph: "D", accent: false },
  { x: 227, y: 300, rotate: -20, glyph: "Á", accent: false },
  { x: 278, y: 326, rotate: -11, glyph: "I", accent: true },
  { x: 482, y: 326, rotate: 11, glyph: "L", accent: false },
  { x: 533, y: 300, rotate: 20, glyph: "É", accent: false },
  { x: 576, y: 265, rotate: 28, glyph: "I", accent: false },
  { x: 139, y: 313, rotate: -33, glyph: "R", accent: false },
  { x: 621, y: 313, rotate: 33, glyph: "E", accent: false },
  { x: 211, y: 410, rotate: -20, glyph: "A", accent: false },
  { x: 549, y: 410, rotate: 20, glyph: "N", accent: true },
];

const seatBlocks = [
  { x: 130, y: 229, rotate: -32 },
  { x: 163, y: 253, rotate: -28 },
  { x: 200, y: 276, rotate: -23 },
  { x: 240, y: 296, rotate: -17 },
  { x: 284, y: 309, rotate: -10 },
  { x: 330, y: 317, rotate: -4 },
  { x: 430, y: 317, rotate: 4 },
  { x: 476, y: 309, rotate: 10 },
  { x: 520, y: 296, rotate: 17 },
  { x: 560, y: 276, rotate: 23 },
  { x: 597, y: 253, rotate: 28 },
  { x: 630, y: 229, rotate: 32 },
];

function stageOpacity(stage: TraceStage, target: Exclude<TraceStage, "idle">) {
  if (stage === target) return 1;
  if (target === "speaker" && (stage === "attention" || stage === "prediction")) return 0.38;
  if (target === "attention" && stage === "prediction") return 0.28;
  return 0.1;
}

export function SceneFallback({
  visible = true,
  traceStage,
  traceMotion,
  traceRunId,
  running,
}: SceneFallbackProps) {
  const transition = traceMotion === "animate" ? "opacity 220ms cubic-bezier(.2,.8,.2,1)" : "none";

  return (
    <div
      className="scene-fallback"
      aria-hidden="true"
      data-poster-visible={visible ? "true" : "false"}
      data-trace-stage={traceStage}
      data-trace-motion={traceMotion}
      data-trace-running={running ? "true" : "false"}
      data-trace-run-id={traceRunId}
    >
      <svg
        viewBox="0 0 760 620"
        width="100%"
        height="100%"
        preserveAspectRatio="xMidYMid meet"
        focusable="false"
        style={{ display: "block", width: "100%", height: "100%" }}
      >
        <defs>
          <linearGradient id="poster-floor" x1="0" y1="0" x2="0.85" y2="1">
            <stop offset="0" stopColor="#153d31" />
            <stop offset="0.52" stopColor="#0c241d" />
            <stop offset="1" stopColor="#07110e" />
          </linearGradient>
          <linearGradient id="poster-timber" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#2b2a20" />
            <stop offset="0.45" stopColor="#171b16" />
            <stop offset="1" stopColor="#090e0c" />
          </linearGradient>
          <linearGradient id="poster-upholstery" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#2c8b68" />
            <stop offset="0.48" stopColor="#176044" />
            <stop offset="1" stopColor="#0f3529" />
          </linearGradient>
          <linearGradient id="poster-brass" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#f1d58f" />
            <stop offset="0.42" stopColor="#c9a55c" />
            <stop offset="1" stopColor="#72572a" />
          </linearGradient>
          <radialGradient id="poster-light" cx="50%" cy="42%" r="54%">
            <stop offset="0" stopColor="#2c8b68" stopOpacity="0.2" />
            <stop offset="0.43" stopColor="#123a2d" stopOpacity="0.09" />
            <stop offset="1" stopColor="#07110e" stopOpacity="0" />
          </radialGradient>
          <filter id="poster-shadow" x="-30%" y="-40%" width="160%" height="200%">
            <feDropShadow dx="0" dy="15" stdDeviation="16" floodColor="#000000" floodOpacity="0.44" />
          </filter>
          <filter id="poster-soft-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <clipPath id="poster-frame">
            <rect x="28" y="24" width="704" height="568" rx="30" />
          </clipPath>
        </defs>

        <g clipPath="url(#poster-frame)">
          <ellipse cx="404" cy="294" rx="334" ry="288" fill="url(#poster-light)" />
          <g data-chamber-part="plinth" filter="url(#poster-shadow)">
            <ellipse cx="380" cy="518" rx="286" ry="58" fill="#030806" opacity="0.7" />
            <path d="M104 493 C188 543 572 558 656 493 L632 526 C527 577 226 570 128 526 Z" fill="url(#poster-timber)" stroke="#c9a55c" strokeOpacity="0.11" />
            <path d="M130 493 C223 532 541 539 630 493" fill="none" stroke="#91a39c" strokeOpacity="0.13" />
          </g>
          <path d="M65 542 C174 580 578 592 704 522" fill="none" stroke="#91a39c" strokeOpacity="0.06" />
          <path d="M90 558 C235 596 566 594 692 539" fill="none" stroke="#c9a55c" strokeOpacity="0.08" />

          <g filter="url(#poster-shadow)">
            <path d="M110 217 C82 334 150 477 380 538 C610 477 678 334 650 217 L615 203 C631 307 568 419 380 472 C192 419 129 307 145 203 Z" fill="#050b09" opacity="0.78" />
            <path d="M137 204 C114 312 178 433 380 493 C582 433 646 312 623 204 C565 144 195 144 137 204 Z" fill="url(#poster-floor)" stroke="#406859" strokeOpacity="0.38" />

            {/* Four quiet bands refer to the model's four decoder layers. */}
            {[0, 1, 2, 3].map((layer) => (
              <path
                key={layer}
                d={`M ${240 + layer * 17} ${262 + layer * 17} C ${256 + layer * 10} ${338 + layer * 9}, ${504 - layer * 10} ${338 + layer * 9}, ${520 - layer * 17} ${262 + layer * 17}`}
                fill="none"
                stroke={layer === 3 ? "#c9a55c" : "#91a39c"}
                strokeOpacity={layer === 3 ? 0.2 : 0.09}
                strokeWidth="1.2"
              />
            ))}

            {/* Stepped horseshoe benches: dark timber carcasses with green upholstery. */}
            <g fill="none" strokeLinecap="round">
              <path d="M124 192 C92 328 170 486 380 546 C590 486 668 328 636 192" stroke="#050907" strokeWidth="46" />
              <path d="M124 192 C92 328 170 486 380 546 C590 486 668 328 636 192" stroke="url(#poster-timber)" strokeWidth="38" />
              <path d="M124 185 C100 315 177 451 380 512 C583 451 660 315 636 185" stroke="url(#poster-upholstery)" strokeWidth="15" />
              <path d="M124 181 C100 309 179 439 380 500 C581 439 660 309 636 181" stroke="#65b494" strokeOpacity="0.28" strokeWidth="2" />

              <path d="M173 224 C153 323 216 421 380 468 C544 421 607 323 587 224" stroke="#070b09" strokeWidth="42" />
              <path d="M173 224 C153 323 216 421 380 468 C544 421 607 323 587 224" stroke="url(#poster-timber)" strokeWidth="34" />
              <path d="M173 218 C160 309 222 394 380 438 C538 394 600 309 587 218" stroke="url(#poster-upholstery)" strokeWidth="14" />
              <path d="M173 214 C162 302 225 383 380 427 C535 383 598 302 587 214" stroke="#65b494" strokeOpacity="0.25" strokeWidth="2" />

              <path d="M224 252 C214 320 267 374 380 405 C493 374 546 320 536 252" stroke="#080c0a" strokeWidth="38" />
              <path d="M224 252 C214 320 267 374 380 405 C493 374 546 320 536 252" stroke="url(#poster-timber)" strokeWidth="30" />
              <path d="M224 247 C220 306 270 348 380 378 C490 348 540 306 536 247" stroke="url(#poster-upholstery)" strokeWidth="13" />
              <path d="M225 242 C223 297 274 338 380 367 C486 338 537 297 535 242" stroke="#65b494" strokeOpacity="0.24" strokeWidth="2" />
            </g>

            {/* Presiding desk, speaking floor and central aisle. */}
            <path d="M298 171 L462 171 L485 221 L275 221 Z" fill="#080d0b" opacity="0.9" />
            <path d="M303 160 L457 160 L474 202 L286 202 Z" fill="url(#poster-timber)" stroke="#765f35" strokeOpacity="0.45" />
            <path d="M321 151 L439 151 L450 176 L310 176 Z" fill="#123a2d" stroke="#c9a55c" strokeOpacity="0.35" />
            <rect x="333" y="132" width="94" height="24" rx="5" fill="#0b1914" stroke="#91a39c" strokeOpacity="0.18" />
            <path d="M338 132 L422 132 L411 119 L349 119 Z" fill="url(#poster-timber)" />

            <path d="M342 235 L418 235 L443 421 L317 421 Z" fill="#081711" stroke="#91a39c" strokeOpacity="0.12" />
            <path d="M377 251 L383 251 L397 418 L363 418 Z" fill="#c9a55c" opacity="0.11" />
            <ellipse cx="380" cy="269" rx="43" ry="17" fill="#07110e" stroke="#c9a55c" strokeOpacity="0.22" />

            <path d="M365 255 L395 255 L400 292 L360 292 Z" fill="url(#poster-timber)" stroke="#c9a55c" strokeOpacity="0.52" />
            <path d="M372 236 L388 236 L392 260 L368 260 Z" fill="url(#poster-brass)" />
            <ellipse cx="380" cy="235" rx="11" ry="4" fill="#f1d58f" opacity="0.72" />

            {/* Repeated seat blocks establish scale without literal institutional detail. */}
            <g>
              {seatBlocks.map((seat, index) => (
                <g key={index} transform={`translate(${seat.x} ${seat.y}) rotate(${seat.rotate})`}>
                  <rect x="-13" y="-7" width="26" height="14" rx="4" fill="#07110e" opacity="0.72" />
                  <rect x="-11" y="-9" width="22" height="11" rx="3" fill={index % 3 === 0 ? "#287d5e" : "#18533e"} stroke="#7bb89f" strokeOpacity="0.18" />
                </g>
              ))}
            </g>

            {/* Character tiles are attached to speaking places rather than floating. */}
            <g fontFamily="Newsreader, Georgia, serif" fontSize="12" textAnchor="middle">
              {seatTiles.map((tile, index) => (
                <g key={index} transform={`translate(${tile.x} ${tile.y}) rotate(${tile.rotate})`}>
                  <rect x="-13" y="-13" width="26" height="26" rx="6" fill="#0a1914" stroke="#91a39c" strokeOpacity="0.28" />
                  <text y="4" fill="#e9e2d0">{tile.glyph}</text>
                </g>
              ))}
            </g>

            <g
              data-trace-group="speaker"
              opacity={stageOpacity(traceStage, "speaker")}
              style={{ transition }}
              transform="translate(278 326) rotate(-11)"
              fontFamily="Newsreader, Georgia, serif"
              fontSize="12"
              textAnchor="middle"
            >
              <g
                style={{
                  transform: traceMotion === "animate" && traceStage === "speaker"
                    ? "translateY(-7px)"
                    : "translateY(0)",
                  transition: traceMotion === "animate"
                    ? "transform 320ms cubic-bezier(.2,.8,.2,1)"
                    : "none",
                }}
              >
                <circle r="22" fill="#c9a55c" opacity="0.2" filter="url(#poster-soft-glow)" />
                <rect x="-15" y="-15" width="30" height="30" rx="7" fill="url(#poster-brass)" stroke="#f1d58f" strokeOpacity="0.82" />
                <text y="4" fill="#07110e">I</text>
              </g>
            </g>

            {/* Eight attention paths converge from the active speaker to a prediction. */}
            <g
              data-trace-group="attention"
              fill="none"
              strokeLinecap="round"
              opacity={stageOpacity(traceStage, "attention")}
              style={{ transition }}
            >
              {[0, 1, 2, 3, 4, 5, 6, 7].map((head) => {
                const path = `M ${278 + head * 0.7} ${326 - head * 0.4} C ${314 + head * 4} ${288 - head * 2}, ${390 + head * 5} ${292 + head * 2}, 549 410`;
                return (
                  <g key={head}>
                    <path
                      d={path}
                      stroke={head === 2 || head === 6 ? "#c9a55c" : "#91a39c"}
                      strokeOpacity={0.28 + head * 0.025}
                      strokeWidth={head === 6 ? 1.7 : 0.85}
                    />
                    {traceStage === "attention" && traceMotion === "animate" && (
                      <circle key={`${traceRunId}-${head}`} r="2.4" fill="#d8bb75" stroke="none">
                        <animateMotion
                          path={path}
                          begin={`${head * 35}ms`}
                          dur="760ms"
                          repeatCount="1"
                          fill="freeze"
                        />
                      </circle>
                    )}
                  </g>
                );
              })}
              <circle cx="278" cy="326" r="3.5" fill="#f1d58f" stroke="none" />
            </g>

            <g
              data-trace-group="prediction"
              opacity={stageOpacity(traceStage, "prediction")}
              style={{ transition }}
              transform="translate(549 410) rotate(20)"
              fontFamily="Newsreader, Georgia, serif"
              fontSize="12"
              textAnchor="middle"
            >
              <circle r="23" fill="#c9a55c" opacity="0.22" filter="url(#poster-soft-glow)" />
              <rect x="-15" y="-15" width="30" height="30" rx="7" fill="url(#poster-brass)" stroke="#f1d58f" strokeOpacity="0.88" />
              <text y="4" fill="#07110e">N</text>
            </g>
          </g>

          <path d="M100 155 L166 155" stroke="#c9a55c" strokeOpacity="0.48" />
          <circle cx="176" cy="155" r="2" fill="#c9a55c" />
          <text x="100" y="141" fill="#91a39c" opacity="0.65" fontFamily="Manrope, Arial, sans-serif" fontSize="9" letterSpacing="2.2">CHAMBER / CHARACTER MODEL</text>
        </g>
      </svg>
    </div>
  );
}
