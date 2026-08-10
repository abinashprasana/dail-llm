import { gzipSync } from "node:zlib";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distRoot = path.join(projectRoot, "dist");
const assetsRoot = path.join(distRoot, "assets");

const budgets = {
  "initial JavaScript": { raw: 425_000, gzip: 140_000 },
  "chamber JavaScript": { raw: 900_000, gzip: 245_000 },
  "application CSS": { raw: 40_000, gzip: 10_000 },
};

function assetPathFromHtml(html, expression, label) {
  const match = html.match(expression);
  if (!match) throw new Error(`Could not find ${label} in dist/index.html.`);
  return path.join(distRoot, match[1].replace(/^\//, ""));
}

async function measure(filePath) {
  const contents = await readFile(filePath);
  return { raw: contents.byteLength, gzip: gzipSync(contents).byteLength };
}

function formatBytes(value) {
  return `${value.toLocaleString("en-US")} B`;
}

async function main() {
  const html = await readFile(path.join(distRoot, "index.html"), "utf8");
  const assetNames = await readdir(assetsRoot);
  const chamberAssets = assetNames.filter((name) => /^ChamberCanvas-[\w-]+\.js$/.test(name));
  if (chamberAssets.length !== 1) {
    throw new Error(`Expected one lazy ChamberCanvas JavaScript asset, found ${chamberAssets.length}.`);
  }

  const files = {
    "initial JavaScript": assetPathFromHtml(
      html,
      /<script[^>]+type="module"[^>]+src="([^"]+\.js)"/,
      "the initial JavaScript asset",
    ),
    "chamber JavaScript": path.join(assetsRoot, chamberAssets[0]),
    "application CSS": assetPathFromHtml(
      html,
      /<link[^>]+rel="stylesheet"[^>]+href="([^"]+\.css)"/,
      "the application stylesheet",
    ),
  };

  let failed = false;
  for (const [label, filePath] of Object.entries(files)) {
    const size = await measure(filePath);
    const budget = budgets[label];
    const rawPass = size.raw <= budget.raw;
    const gzipPass = size.gzip <= budget.gzip;
    failed ||= !rawPass || !gzipPass;
    const status = rawPass && gzipPass ? "PASS" : "FAIL";
    console.log(
      `${status} ${label}: raw ${formatBytes(size.raw)} / ${formatBytes(budget.raw)}, ` +
      `gzip ${formatBytes(size.gzip)} / ${formatBytes(budget.gzip)}`,
    );
  }

  if (failed) process.exitCode = 1;
}

main().catch((error) => {
  console.error(`Bundle budget check failed: ${error.message}`);
  process.exitCode = 1;
});
