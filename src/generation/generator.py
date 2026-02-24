from src.config import OPENAI_API_KEY


def build_prompt(query: str, context_docs: list[dict]) -> str:
    """Construct a prompt with retrieved context and citation instructions."""
    context_parts = []
    for i, doc in enumerate(context_docs):
        source = doc["metadata"].get("source", "unknown")
        page = doc["metadata"].get("page", "?")
        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{doc['content']}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this."
Always cite your sources using [Source N] format.

Context:
{context_str}

Question: {query}

Answer (with citations):"""


class Generator:
    """Generate answers using LLM with retrieved context."""

    def __init__(self):
        if OPENAI_API_KEY:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.mode = "openai"
        else:
            self.client = None
            self.mode = "local"

    def generate(self, query: str, context_docs: list[dict]) -> dict:
        prompt = build_prompt(query, context_docs)

        if self.mode == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
        else:
            answer = self._extractive_fallback(query, context_docs)

        return {
            "answer": answer,
            "sources": [
                {
                    "source": d["metadata"].get("source"),
                    "page": d["metadata"].get("page"),
                    "score": d["similarity_score"],
                    "excerpt": d["content"][:200]
                }
                for d in context_docs
            ],
            "model": self.mode
        }

    def _extractive_fallback(self, query: str, docs: list[dict]) -> str:
        """Simple fallback when no API key is available."""
        top = docs[0] if docs else None
        if not top:
            return "No relevant documents found."
        source = top["metadata"].get("source", "unknown")
        page = top["metadata"].get("page", "?")
        return (
            f"Based on the most relevant passage [Source: {source}, Page {page}]:\n\n"
            f"{top['content'][:500]}"
        )