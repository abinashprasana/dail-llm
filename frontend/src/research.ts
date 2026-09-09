export const researchRoot = "/research-data/pilot";
export const policyNames = {
  uniform: "Uniform positions",
  speech_balanced: "Across speeches",
  context_diverse: "Diverse contexts",
};
export type Policy = keyof typeof policyNames;
export type Metrics = {
  targets: number;
  supported: number;
  unknown: number;
  supported_target_bpc: number | null;
  coverage: number | null;
  mapped_token_loss: number | null;
};
export type Provenance = {
  binding: Record<string, string>;
  checkpoint_sha256: string;
  tokenizer_sha256: string;
  report_sha256: string;
};
export type Neighbour = {
  speech_id: string;
  date: string;
  member: string;
  title: string;
  offset: number;
  span_start: number;
  source_span: string;
  memory_mass: number;
  mixture_mass: number;
};
export type Distribution = {
  character: string;
  base: number;
  before: number;
  after: number;
  delta: number;
};
export type Inspection = {
  schema_version: number;
  release_id: string;
  policy: Policy;
  prefix: string;
  excluded_speech: string | null;
  parameters: { k: number; weight: number; temperature: number };
  remaining_entries: number;
  removed_entries: number;
  unknown_characters: number;
  neighbours_before: Neighbour[];
  neighbours_after: Neighbour[];
  distributions: Distribution[];
  provenance: Provenance;
};
export type Recording = {
  original: Inspection;
  source_entries: Record<string, Neighbour>;
  exclusions: Record<
    string,
    {
      excluded_speech: string;
      remaining_entries: number;
      removed_entries: number;
      after: number[];
      neighbours: [string, number][];
    }
  >;
};
export type Example = {
  id: string;
  prefix: string;
  kind: string;
  files: Record<Policy, { name: string; sha256: string }>;
};
export type Split = {
  aggregate: Metrics;
  continuing_speakers?: Metrics;
  matched?: Metrics;
  perturbations: Record<
    string,
    {
      masked_characters: number;
      masked_input: Metrics;
      same_target_control: Metrics;
    }
  >;
};
export type Summary = {
  schema_version: number;
  release_id: string;
  label: string;
  seed: number;
  steps: number;
  batch_size: number;
  provenance: Provenance;
  periods: Record<string, string[]>;
  selection: Record<
    string,
    { speeches: number; characters: number; selected_groups: number }
  >;
  counts: Record<string, number>;
  source: string;
  source_sha256: string;
  history: Record<
    string,
    { parameters: number; splits: Record<string, Split> }
  >;
  memory: Record<
    Policy,
    {
      metrics: Metrics;
      entries: number;
      parameters: { k: number; weight: number; temperature: number };
      slices: Record<
        string,
        { spans: number; bpc: number | null; accuracy: number | null }
      >;
      retrieval: { ms_per_query: number };
      copying: number | null;
      tokens_per_second: number | null;
      comparison: { difference_bpc: number; interval: number[] | null } | null;
    }
  >;
  cohort: {
    pairs: number;
    unmatched_earlier: number;
    unmatched_later: number;
    continuing_speakers: number;
    continuing_speeches: Record<string, number>;
  };
  verification: { status: string; checks: number; sha256: string };
  references: [string, string][];
  training_history: {
    step: number;
    train_loss: number;
    validation_loss: number;
  }[];
  execution: { elapsed_seconds: number; peak_memory_bytes: number };
  examples: Example[];
};
export type Capabilities = {
  live: boolean;
  release_id: string | null;
  reason: string;
};

export const number = (value: number | null | undefined, digits = 4) =>
  value == null ? "Unavailable" : value.toFixed(digits);
export const characterLabel = (value: string) =>
  ({
    " ": "Space",
    "\n": "Newline",
    "\t": "Tab",
    "<unk>": "Unknown",
    "<pad>": "Padding",
  })[value] ?? value;

export function sourceSegments(
  source: Pick<Neighbour, "source_span" | "offset" | "span_start">,
) {
  const characters = Array.from(source.source_span);
  const index = source.offset - source.span_start;
  return {
    before: characters.slice(0, index).join(""),
    target: characters[index] ?? "",
    after: characters.slice(index + 1).join(""),
  };
}

export function recordedResult(
  recording: Recording,
  excluded: string | null,
): Inspection {
  const original = recording.original;
  if (!excluded) return original;
  const change = recording.exclusions[excluded];
  if (!change || change.after.length !== original.distributions.length)
    throw new Error("This source removal is unavailable.");
  return {
    ...original,
    ...change,
    distributions: original.distributions.map((d, i) => ({
      ...d,
      after: change.after[i],
      delta: change.after[i] - d.before,
    })),
    neighbours_after: change.neighbours.map(([key, mass]) => ({
      ...recording.source_entries[key],
      memory_mass: mass,
      mixture_mass: mass * original.parameters.weight,
    })),
  };
}

export function visibleDistributions(rows: Distribution[]) {
  const keys = ["base", "before", "after"] as const;
  const selected = new Set(
    keys.flatMap((key) =>
      [...rows]
        .sort((a, b) => b[key] - a[key])
        .slice(0, 8)
        .map((d) => d.character),
    ),
  );
  return rows
    .filter((d) => selected.has(d.character))
    .sort(
      (a, b) =>
        Math.max(b.base, b.before, b.after) -
          Math.max(a.base, a.before, a.after) ||
        a.character.localeCompare(b.character),
    );
}

export async function fetchJSON<T>(
  url: string,
  signal?: AbortSignal,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(url, { ...init, signal });
  if (!response.ok) {
    if (response.status === 429)
      throw new Error(
        `The model is busy. Try again in ${response.headers.get("Retry-After") || "a few"} seconds.`,
      );
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "The research data could not be loaded. Please try again.",
    );
  }
  return response.json() as Promise<T>;
}

export async function fetchRecording(
  example: Example,
  policy: Policy,
  summary: Summary,
  signal: AbortSignal,
) {
  const file = example.files[policy];
  if (
    !/^example-\d+-(uniform|speech_balanced|context_diverse)\.json$/.test(
      file.name,
    )
  )
    throw new Error("Invalid example reference.");
  const response = await fetch(`${researchRoot}/${file.name}`, { signal });
  if (!response.ok) throw new Error("This recorded example is unavailable.");
  const bytes = await response.arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  if (hash !== file.sha256)
    throw new Error("The example does not match this research release.");
  const value = JSON.parse(new TextDecoder().decode(bytes)) as Recording;
  if (
    value.original.release_id !== summary.release_id ||
    value.original.policy !== policy ||
    value.original.prefix !== example.prefix
  )
    throw new Error("The example belongs to a different research release.");
  return value;
}
