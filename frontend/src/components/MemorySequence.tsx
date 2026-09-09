import { ArrowRight } from "lucide-react";
import {
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
} from "motion/react";
import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import "../memory-sequence.css";

const steps = [
  {
    title: "Start with a speech",
    text: "The archive preserves the speech, its date, and its speaker. Each source remains identifiable throughout the experiment.",
  },
  {
    title: "Keep a small memory",
    text: "Selected positions become 256-dimensional vectors, paired with the next character. Only training speeches enter this memory.",
  },
  {
    title: "Find nearby contexts",
    text: "For a fixed prefix, exact search finds the nearest stored vectors. Their next characters contribute to a probability distribution.",
  },
  {
    title: "Remove a source. Search again.",
    text: "Exclude a speech and every one of its entries leaves the search. Compare the new distribution with the original prediction.",
  },
];

function MemoryDrawing({ stage }: { stage: number }) {
  return (
    <svg viewBox="0 0 500 430" className="memory-drawing" aria-hidden="true">
      <path
        d="M65 310 245 403 455 296"
        fill="none"
        stroke="#c9a55c"
        strokeOpacity=".15"
      />
      {[0, 1, 2].map((sheet) => (
        <g
          key={sheet}
          className={`memory-sheet ${stage === 3 && sheet === 1 ? "removed-sheet" : ""}`}
          transform={`translate(${80 + sheet * 38} ${125 - sheet * 25})`}
        >
          <path
            d="m0 70 155-80 154 79-155 81z"
            fill={sheet === 2 ? "#163c2d" : "#0e281e"}
            stroke="#91a39c"
            strokeOpacity=".4"
          />
          <path
            d="m0 70 154 80 155-81v8l-155 80L0 78z"
            fill="#07110e"
            stroke="#91a39c"
            strokeOpacity=".22"
          />
          {[0, 1, 2, 3].map((i) => (
            <path
              key={i}
              d={`m${56 + i * 17} ${55 + i * 9} 99-51`}
              stroke={stage >= 1 ? "#c9a55c" : "#91a39c"}
              strokeWidth="2"
              strokeOpacity={0.6 - i * 0.09}
            />
          ))}
        </g>
      ))}
      <g opacity={stage >= 2 ? 1 : 0.2} className="memory-connections">
        <path
          d="M298 190C410 220 420 282 348 319M250 236C330 260 320 305 348 319M195 285C228 338 300 359 348 319"
          fill="none"
          stroke="#c9a55c"
          strokeWidth="1.5"
          strokeDasharray={stage === 3 ? "4 5" : undefined}
        />
        <path d="m348 297 30 16v31l-30 16-30-16v-31z" fill="#c9a55c" />
        <text
          x="348"
          y="338"
          textAnchor="middle"
          fill="#07110e"
          fontFamily="Newsreader, Georgia, serif"
          fontSize="27"
        >
          a
        </text>
      </g>
      <text
        x="45"
        y="382"
        fill="#91a39c"
        fontFamily="Manrope, sans-serif"
        fontSize="10"
        letterSpacing="2"
      >
        {
          [
            "SOURCE SPEECH",
            "TRAINING MEMORY",
            "NEAREST CONTEXTS",
            "SOURCE REMOVED",
          ][stage]
        }
      </text>
    </svg>
  );
}

export function MemorySequence() {
  const ref = useRef<HTMLElement>(null);
  const [stage, setStage] = useState(0);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start center", "end center"],
  });
  useMotionValueEvent(scrollYProgress, "change", (value) =>
    setStage(Math.min(3, Math.floor(value * 4))),
  );
  return (
    <section
      className={`memory-sequence content-section ${reduced ? "is-reduced" : ""}`}
      ref={ref}
      aria-labelledby="memory-sequence-title"
    >
      <div className="page-width">
        <div className="split-heading">
          <div>
            <div className="eyebrow">
              <span /> Speech memory
            </div>
            <h2 id="memory-sequence-title">Follow a speech into memory</h2>
          </div>
          <p>
            A prediction can be inspected at its sources. This illustration
            follows the path; the Research page shows the measured results.
          </p>
        </div>
        <div className="memory-story">
          <figure className="memory-sticky">
            <MemoryDrawing stage={reduced ? 2 : stage} />
            <figcaption>
              Illustration of retrieval, not a live model trace.
            </figcaption>
            <div className="memory-story-progress" aria-hidden="true">
              {steps.map((_, i) => (
                <i className={i === stage ? "is-active" : ""} key={i} />
              ))}
            </div>
          </figure>
          <div className="memory-story-steps">
            {steps.map((step, i) => (
              <motion.article
                key={step.title}
                className={stage === i ? "is-active" : ""}
                initial={{ opacity: 1 }}
              >
                <span className="memory-step-number">0{i + 1}</span>
                <div className="memory-mobile-drawing">
                  <MemoryDrawing stage={i} />
                </div>
                <h3>{step.title}</h3>
                <p>{step.text}</p>
                {i === 3 && (
                  <Link className="text-link" to="/research?view=memory">
                    Inspect a recorded prediction <ArrowRight size={16} />
                  </Link>
                )}
              </motion.article>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
