export interface DatasetManifest {
  schema_version: number;
  dataset?: {
    name?: string;
    citation?: string;
    doi?: string;
    full_date_range?: string;
  };
  corpus?: {
    minimum_year?: number;
    selected_date_min?: string | null;
    selected_date_max?: string | null;
    accepted_speeches?: number;
    clean_bytes?: number;
    clean_characters?: number;
    sha256?: string;
  };
  processing?: {
    chunk_count?: number;
    split_characters?: {
      train: number;
      validation: number;
      test: number;
    };
  };
}

export interface ModelMetadata {
  name: string;
  checkpoint: { name: string; sha256: string };
  architecture: {
    block_size: number;
    embed_dim: number;
    n_layers: number;
    n_heads: number;
    dropout: number;
    vocab_size: number;
    parameters: number;
    type: string;
  };
  dataset: DatasetManifest | null;
}

export interface HealthStatus {
  status: "ready" | "degraded";
  version: string;
  model_loaded: boolean;
  device: string;
  error?: string | null;
}

export interface GenerationResult {
  text: string;
  prompt: string;
  generated_characters: number;
  elapsed_ms: number;
  filtered_characters: string[];
}

export interface EvaluationReport {
  generated_at: string;
  checkpoint: {
    name: string;
    parameters: number;
    config: Record<string, number>;
  };
  metrics: {
    cross_entropy: number | null;
    perplexity: number | null;
    bits_per_character: number | null;
    next_character_accuracy: number | null;
  };
  samples: Array<{
    prompt: string;
    text: string;
    continuation: string;
    repeated_word_trigram_rate: number | null;
  }>;
}

export interface AttentionResult {
  prompt: string;
  labels: string[];
  layer: number;
  head: number | null;
  matrices: number[][][];
  filtered_characters: string[];
}
