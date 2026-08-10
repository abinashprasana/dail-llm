import { motion } from "motion/react";

export function MetricCard({ label, value, note, index = 0 }: { label: string; value: string; note: string; index?: number }) {
  return (
    <motion.article
      className="metric-card"
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.55 }}
      transition={{ delay: index * 0.08, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{note}</p>
    </motion.article>
  );
}
