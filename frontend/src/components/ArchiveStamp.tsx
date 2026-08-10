import { motion } from "motion/react";

const ease = [0.22, 1, 0.36, 1] as const;

export function ArchiveStamp({ reducedMotion }: { reducedMotion: boolean }) {
  const impression = reducedMotion
    ? { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { duration: 0.18 } } }
    : { hidden: { opacity: 1 }, visible: { opacity: 1 } };
  const ink = reducedMotion
    ? { hidden: { opacity: 1 }, visible: { opacity: 1 } }
    : {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { delay: 0.5, duration: 0.28, ease } },
      };
  const border = reducedMotion
    ? { hidden: { opacity: 1, pathLength: 1 }, visible: { opacity: 1, pathLength: 1 } }
    : {
        hidden: { opacity: 0, pathLength: 0 },
        visible: { opacity: 1, pathLength: 1, transition: { delay: 0.5, duration: 0.28, ease } },
      };

  return (
    <div
      className="archive-stamp"
      role="note"
      aria-label="Dataset source: Harvard Dataverse, record DVN/6MZN76."
    >
      <motion.div
        className="archive-stamp-sequence"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.65 }}
        variants={{ hidden: {}, visible: {} }}
      >
        <motion.div className="archive-stamp-impression" variants={impression} aria-hidden="true">
          <svg viewBox="0 0 176 62" preserveAspectRatio="none" focusable="false" aria-hidden="true">
            <motion.path
              className="archive-stamp-border"
              d="M5 5 170 3l2 52L6 58 5 5Z"
              fill="none"
              variants={border}
            />
            <motion.g className="archive-stamp-defects" variants={ink}>
              <path d="M18 5h19m52-1h28m35 0h9M7 20v12m0 15v5m25 5h17m67-1h21m21-1h9" />
              <circle cx="15" cy="54" r="1.2" />
              <circle cx="165" cy="13" r="1" />
            </motion.g>
          </svg>
          <motion.span className="archive-stamp-copy" variants={ink}>
            <span>Harvard Dataverse</span>
            <span>DVN/6MZN76</span>
          </motion.span>
        </motion.div>
        {!reducedMotion && (
          <motion.div
            className="archive-stamp-press"
            aria-hidden="true"
            variants={{
              hidden: { x: 36, y: 28, scaleY: 1, opacity: 0 },
              visible: {
                x: [36, 0, -2, 0, 0, -18],
                y: [28, 0, 2, 9, 9, -30],
                scaleY: [1, 1, 1, 0.985, 0.985, 1],
                opacity: [0, 1, 1, 1, 1, 0],
                transition: {
                  duration: 1.08,
                  times: [0, 0.2963, 0.4259, 0.5185, 0.5741, 1],
                  ease,
                },
              },
            }}
          >
            <motion.span
              className="archive-stamp-press-shadow"
              variants={{
                hidden: { opacity: 0, scaleX: 1.12 },
                visible: {
                  opacity: [0, 0.35, 0.35, 0.62, 0.4, 0],
                  scaleX: [1.12, 1, 1, 0.72, 0.84, 1],
                  transition: {
                    duration: 1.08,
                    times: [0, 0.2963, 0.4259, 0.5185, 0.5741, 1],
                    ease,
                  },
                },
              }}
            />
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
