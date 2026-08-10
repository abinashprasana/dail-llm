"""
CLI script to generate text with the Dáil LLM (legacy src/ package).
"""
import argparse
from src.inference import ModelWrapper
from src.rag.retriever import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True, help="Prompt text")
    ap.add_argument("--top_k", type=int, default=3, help="Number of retrieved chunks (0 to disable RAG)")
    ap.add_argument("--max_new_tokens", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    # Load Model
    print("Initializing model...")
    wrapper = ModelWrapper()

    final_prompt = args.prompt

    # RAG Step
    if args.top_k > 0:
        print(f"Retrieving top {args.top_k} chunks...")
        rag = RAGPipeline()
        hits = rag.search(args.prompt, top_k=args.top_k)

        print("\n=== RETRIEVED CHUNKS ===\n")
        retrieved_texts = []
        for i, score, txt in hits:
            print(f"- [Chunk {i}] (score={score:.4f}): {txt[:100]}...")
            retrieved_texts.append(txt)

        context_str = "\n\n---\n\n".join(retrieved_texts)
        final_prompt = RAGPipeline.format_augmented_prompt(args.prompt, context_str)

    print("\n=== GENERATING ===\n")
    output = wrapper.generate(final_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    print("\n=== FINAL OUTPUT ===\n")
    print(output)


if __name__ == "__main__":
    main()
