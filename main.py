from RagPipeline import RAGPipeline
from RAGASEvaluator import RAGASEvaluator
from sentence_transformers import CrossEncoder
import google.generativeai as genai
import os

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

def main() -> None:
    file_path = "StateFarm_Final_Estimate.pdf"
    rag = RAGPipeline()
    collection="InsuranceEstimate"
    #rag.ingest(file_path,collection)
    query = "actual cost total floor covering ceramic tile"
    results = rag.search(query,collection)
    for res in results:
        print(f"\n--- MATCH (Score: {res['score']:.4f}) ---")
        print(f"Detail Found: {res['child_match']}...")
        print(f"Context Provided to LLM: {res['full_context']}...")

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidate_texts = [hit["child_match"] for hit in results]

    pairs = [[query, doc] for doc in candidate_texts]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_texts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = ranked[:5]
    for doc, score in ranked:
        print("\n--- RERANKED ---")
        print("Score:", score)
        print("Text:", doc)
    
    top_docs = [doc for doc, score in ranked[:3]]

    context = "\n\n".join(top_docs)

    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Answer the user's question ONLY using the provided context.

    Question:
    {query}

    Context:
    {context}

    If the answer is not in the context, say "Not found in document."
    """
    print("\n=== CONTEXT SENT TO LLM ===\n")
    print(context)
    response = model.generate_content(prompt)

    print("\n=== FINAL ANSWER ===")
    print(response.text)

    evaluator = RAGASEvaluator()
    ground_truth = [
        "The actual cost total floor covering ceramic tile is $12,859.43"
    ]
    scores = evaluator.evaluate_rag(
        query,
        response.text,
        top_docs,
        ground_truth[0]
    )

    print(scores)

if __name__ == "__main__":
    main()
