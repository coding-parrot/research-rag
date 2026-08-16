"""How do we know it works? Ask questions we know the answers to.

Two numbers per question:
  retrieved - did the right paper come back in the top-k?  (retrieval quality)
  cited     - did the final answer cite that paper, with a verified quote?
              (end-to-end groundedness)

Retrieval is the ceiling: if the right chunk never comes back, the model cannot
cite it. Run this after any change to chunking, embedding, or prompts.
"""

from rag import store
from rag.pipeline import ask

# question -> the paper id a correct answer must draw on
GOLDEN = {
    "How does LoRA reduce the number of trainable parameters?": "lora",
    "What data type does QLoRA introduce for quantizing weights?": "qlora",
    "How does the Switch Transformer route tokens to experts?": "switch_transformer",
    "What is the selection mechanism in Mamba?": "mamba",
    "What bottleneck does FlashAttention target?": "flash_attention",
    "What are the three training steps of InstructGPT?": "instructgpt",
    "How does PagedAttention manage the KV cache?": "paged_attention",
    "Why does multi-query attention speed up decoding?": "mqa",
    "What is scaled dot-product attention?": "attention",
    "What role does temperature play in knowledge distillation?": "distillation",
}


def main() -> None:
    index, chunks = store.load()
    retrieved_hits = cited_hits = 0

    print(f"{'question':58s} retrieved  cited")
    for question, paper in GOLDEN.items():
        results = store.search(index, chunks, question)
        retrieved = any(chunk.paper == paper for chunk, _ in results)

        answer = ask(question)
        cited = any(paper in c["chunk_id"].split(":")[0] for c in answer.citations)

        retrieved_hits += retrieved
        cited_hits += cited
        print(f"{question[:57]:58s} {'yes' if retrieved else 'NO ':9s}  {'yes' if cited else 'NO'}")

    n = len(GOLDEN)
    print(f"\nretrieval hit rate: {retrieved_hits}/{n}    grounded-citation rate: {cited_hits}/{n}")


if __name__ == "__main__":
    main()
