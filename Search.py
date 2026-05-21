
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseVector
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseIndexParams
import uuid
class HybridSearch:
    def __init__(self):
       pass 
    def retrieve_with_context(self, query_text, collection_name, top_k=3):
        client = QdrantClient(host="localhost", port=6333)

        # 1. Search only the children for the most relevant specific detail
        search_results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Dense leg — semantic
            Prefetch(
                query=models.Document(text=query_text, model="BAAI/bge-large-en-v1.5"),
                using="fast-bge-large-en",
                filter=models.Filter(
                    must=[models.FieldCondition(key="is_child", match=models.MatchValue(value=True))]
                ),
                limit=top_k * 3   # fetch more before fusion
            ),
            # Sparse leg — keyword BM25
            Prefetch(
                query=models.Document(text=query_text, model="Qdrant/bm25"),
                using="bm25",
                filter=models.Filter(
                    must=[models.FieldCondition(key="is_child", match=models.MatchValue(value=True))]
                ),
                limit=top_k * 3
            ),
        ],
        # RRF merges both ranked lists into one
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k
        ).points

        final_results = []

        for hit in search_results:
            parent_id = hit.payload["metadata"].get("parentId")
            
            # 2. Fetch the Parent context for this child
            if parent_id:
                parent_record = client.retrieve(
                    collection_name=collection_name,
                    ids=[parent_id]
                )
                
                if parent_record:
                    parent_text = parent_record[0].payload.get("text")
                    final_results.append({
                        "child_match": hit.payload.get("text"),
                        "full_context": parent_text,
                        "score": hit.score
                    })

        return final_results