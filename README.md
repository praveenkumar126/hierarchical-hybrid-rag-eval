# ragSystem
# Hierarchical Hybrid RAG System with Automated Ragas Evaluation

An enterprise-grade Retrieval-Augmented Generation (RAG) system built to parse complex documents, execute high-precision hybrid retrieval strategies, and systematically measure performance without relying on manual vibe checks. 

This pipeline features a **Hierarchical Child-to-Parent chunking design** backed by an advanced vector store configuration in Qdrant, optimizing context injection before passing records downstream to a Generative Model.

---

## 🏗️ Architecture Overview

The pipeline breaks down into three distinct operational layers:

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/3dafafe7-5076-41e1-9e7c-f9ed40a4236a" />


1. **Ingestion & Processing Layer:** Extracts layouts smoothly using Markdown schemas, segregating text into parent-child structural relationships.
2. **Hybrid Retrieval Layer:** Simultaneously fires dense semantic vectors and sparse keyword models across deep payload properties, combining them cleanly via Reciprocal Rank Fusion (RRF) and filtering out background noise with a cross-encoder reranker.
3. **Automated Evaluation Layer:** Hooks directly into RAGAS using Gemini as an evaluator judge to analyze production quality scores deterministically.

---

## 📂 Project Structure

* `Ingest.py`: Leverages `Docling` to natively convert structured assets (such as tables or complex form layouts found in PDFs) into strict Markdown layouts.
* `Chunking.py`: Drives structural parsing. Formulates broad "Parent Chunks" to serve as complete contexts, then splits them into localized "Child Chunks" embedded with their respective sections' header titles.
* `StoreChunk.py`: Bootstraps the Qdrant database collection, configuring dense spatial configurations alongside sparse indexing trackers, and handles nested relational uploads.
* `Search.py`: Multi-stage search orchestration executing parallel sub-queries, running RRF across structural points, and restoring structural parent frames before handing payload details off.
* `RagPipeline.py`: Structural facade organizing ingestion and retrieval routines cleanly.
* `RAGASEvaluator.py`: Harnesses `ragas` using `gemini-2.5-flash` and `text-embedding-004` to compute continuous execution benchmarks.
* `main.py`: Operational workflow script that coordinates file initialization, parameters tuning, and output evaluation logging.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* Qdrant instance running locally (`docker run -p 6333:6333 qdrant/qdrant`)

### Step 1: Install Dependencies
```bash
pip install qdrant-client ragas datasets langchain-core langchain-experimental \
            langchain-huggingface langchain-google-genai sentence-transformers docling

Step 2: Configure Environment VariablesExport your credentials securely to access the external inference steps:Bashexport GOOGLE_API_KEY="your-gemini-api-key-here"
🚀 UsageTo trigger the end-to-end extraction, search, and validation run:Bashpython main.py
Deep Dive: How the Code Works1. Contextual Structural Chunking (Chunking.py)To preserve semantic context in technical lookups, the system isolates high-level section boundaries and applies specialized layout dividers:Python# Context injection rule applied during hierarchical processing
contextual_text = f"Section: {parent_header} \nContent: {child_content_string}"
2. Vector Fusion Querying (Search.py)The system routes explicit instructions natively through parallel sub-conditions, using Reciprocal Rank Fusion to balance dense semantics and sparse keyword weights:Pythonquery=FusionQuery(fusion=Fusion.RRF)
3. Execution Benchmarks (RAGASEvaluator.py)The pipeline runs evaluation traces through four distinct RAGAS metrics to isolate failure points across both retrieval and generation steps:MetricTarget AssessedCore ObjectiveFaithfulnessGenerator vs ContextEliminates hallucinations by ensuring statements are grounded in documents.Answer RelevancyGenerator vs QueryMeasures how directly the answer addresses the user's explicit intent.Context PrecisionRetriever vs QueryAssesses if the reranker successfully prioritized the most relevant items.Context RecallRetriever vs Ground TruthValidates if the retriever fetched all necessary information to solve the query.📈 Monitoring RAGAS OutcomesWhen evaluations finish executing, the scores fall within a 0.0 - 1.0 range. Use these targets to diagnose production quality issues:Low Faithfulness (< 0.80): Your generation model is hallucinating or leaking outside data. Tighten your system prompt instructions in main.py.Low Context Precision (< 0.80): Irrelevant chunks are ranking too high. Tune your Cross-Encoder Reranker threshold metrics or inspect chunk sizes.Low Context Recall (< 0.80): The system missed the reference answers entirely. Review your Docling parsing configuration or adjust your text splitting thresholds.
***

### 💡 Recommendation for Your GitHub Repo
Create a file named `README.md` in the root directory of your project and paste the markdown block above inside it. This provides a clean summary for any developer or stakeholder reviewing your codebase!
