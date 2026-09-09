import { gzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../dist",
);
const manifest = JSON.parse(
  await readFile(path.join(root, ".vite/manifest.json"), "utf8"),
);
function closure(key, found = new Set()) {
  if (found.has(key)) return found;
  if (!manifest[key]) throw new Error(`Missing build entry: ${key}`);
  found.add(key);
  for (const dependency of manifest[key].imports ?? [])
    closure(dependency, found);
  return found;
}
const initial = closure("index.html");
function files(keys, type) {
  return [
    ...new Set(
      [...keys].flatMap((key) =>
        type === "css" ? (manifest[key].css ?? []) : [manifest[key].file],
      ),
    ),
  ];
}
let failed = false;
async function check(label, names, rawLimit, gzipLimit) {
  const contents = await Promise.all(
    names.map((name) => readFile(path.join(root, name))),
  );
  const raw = contents.reduce((sum, data) => sum + data.length, 0);
  const gzip = contents.reduce((sum, data) => sum + gzipSync(data).length, 0);
  const pass = raw <= rawLimit && gzip <= gzipLimit;
  failed ||= !pass;
  console.log(
    `${pass ? "PASS" : "FAIL"} ${label}: ${raw} B raw / ${gzip} B gzip`,
  );
}
await check(
  "initial JavaScript (all static imports)",
  files(initial, "js"),
  425_000,
  140_000,
);
await check("application CSS", files(initial, "css"), 40_000, 10_000);
for (const [key, label, raw, gzip] of [
  ["src/components/ChamberCanvas.tsx", "chamber", 900_000, 245_000],
  ["src/pages/ResearchPage.tsx", "research", 220_000, 70_000],
  ["src/components/MemorySequence.tsx", "memory explanation", 35_000, 12_000],
]) {
  const unique = new Set(
    [...closure(key)].filter((item) => !initial.has(item)),
  );
  await check(`${label} JavaScript`, files(unique, "js"), raw, gzip);
  if (key !== "src/components/ChamberCanvas.tsx")
    await check(`${label} CSS`, files(unique, "css"), 30_000, 8_000);
}
const summaryBytes = await readFile(
  path.join(root, "research-data/pilot/summary.json"),
);
const summary = JSON.parse(summaryBytes);
if (summaryBytes.length > 150_000)
  throw new Error("Research summary exceeds 150 KB");
let maximum = 0;
for (const example of summary.examples) {
  for (const [policy, file] of Object.entries(example.files)) {
    if (
      !/^example-\d+-(uniform|speech_balanced|context_diverse)\.json$/.test(
        file.name,
      )
    )
      throw new Error("Invalid research file name");
    const bytes = await readFile(
      path.join(root, "research-data/pilot", file.name),
    );
    if (createHash("sha256").update(bytes).digest("hex") !== file.sha256)
      throw new Error(`Example hash mismatch: ${file.name}`);
    const data = JSON.parse(bytes);
    if (
      data.original.release_id !== summary.release_id ||
      data.original.policy !== policy ||
      data.original.prefix !== example.prefix
    )
      throw new Error("Mixed research release");
    maximum = Math.max(maximum, bytes.length);
    if (bytes.length > 250_000)
      throw new Error(`Example exceeds 250 KB: ${file.name}`);
  }
}
console.log(
  `PASS research artifacts: summary ${summaryBytes.length} B; largest example ${maximum} B; hashes verified`,
);
if (failed) process.exitCode = 1;
