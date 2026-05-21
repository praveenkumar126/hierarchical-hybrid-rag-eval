from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseVector
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseIndexParams
import uuid

class Store:
    def __init__(self):
        pass
    def initialize_collection(self, client, collection_name):
        # Check if collection exists
        if not client.collection_exists(collection_name):
            print(f"Collection {collection_name} not found. Creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    # Match this to your FastEmbed model (BGE-Small is 384)
                    "fast-bge-large-en": VectorParams(size=1024, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                # Qdrant handles BM25 sparse vectors natively via FastEmbed
                "bm25": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
            )
        else:
            print(f"Collection {collection_name} already exists. Ready to upsert.")

    def store_in_qdrant(self, parent_docs, child_docs, collection_name):
        client = QdrantClient(host="localhost", port=6333)
        self.initialize_collection(client, collection_name)
        # 1. Prepare Points for Children (The ones we search against)
        points = []
        for i, child in enumerate(child_docs):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    # Use FastEmbed directly via the Document wrapper
                    vector={
                        "fast-bge-large-en": models.Document(
                            text=child.page_content,
                            model="BAAI/bge-large-en-v1.5"
                        ),
                        "bm25": models.Document(
                        text=child.page_content,
                        model="Qdrant/bm25"       # built-in Qdrant sparse model
                        )
                    },
                     payload={
                        "text": child.page_content,
                        "raw_text": child.metadata["raw_text"],
                        "metadata": child.metadata,
                        "is_child": True
                    }
                )
            )
            
        # 2. Prepare Points for Parents (Context storage - no vector needed)
        # We store these so we can do a 'Lookup' by ID later
        for parent in parent_docs:
            points.append(
                models.PointStruct(
                    id=parent.metadata["docId"],
                    vector={}, # Empty vector for metadata-only points
                    payload={
                        "text": parent.page_content,
                        "metadata": parent.metadata,
                        "is_child": False
                    }
                )
            )

        # 3. Batch Upsert
        client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded {len(points)} points to Qdrant.")

