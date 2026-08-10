import { motion, useReducedMotion, useScroll, useSpring } from "motion/react";
import { useRef } from "react";

const stages = [
  { index: "01", title: "Characters", detail: "A compact vocabulary maps each supported character to an integer." },
  { index: "02", title: "Context", detail: "Token and learned position embeddings share a 256-character window." },
  { index: "03", title: "Attention", detail: "Four decoder blocks use eight causal attention heads to read the preceding text." },
  { index: "04", title: "Next character", detail: "The final projection estimates the next character, then repeats the process." },
];

export function ArchitectureFlow() {
  const section = useRef<HTMLDivElement>(null);
  const reducedMotion = Boolean(useReducedMotion());
  const { scrollYProgress } = useScroll({ target: section, offset: ["start 0.8", "end 0.35"] });
  const progress = useSpring(scrollYProgress, { stiffness: 95, damping: 25 });

  return (
    <div className="architecture-flow" ref={section}>
      <div className="architecture-rail" aria-hidden="true"><motion.span style={{ scaleY: reducedMotion ? 1 : progress }} /></div>
      {stages.map((stage, position) => (
        <motion.article
          className="architecture-stage"
          key={stage.index}
          initial={{ opacity: 0, y: reducedMotion ? 0 : 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.45 }}
          transition={reducedMotion
            ? { duration: 0.18 }
            : { duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="stage-index">{stage.index}</div>
          <div>
            <h3>{stage.title}</h3>
            <p>{stage.detail}</p>
          </div>
          <div className={`stage-signal signal-${position + 1}`} aria-hidden="true">
            {Array.from({ length: position + 3 }, (_, index) => <i key={index} />)}
          </div>
        </motion.article>
      ))}
    </div>
  );
}
