from Ingest import loader
from Chunking import Chunking
from StoreChunk import Store
from Search import HybridSearch
import uuid
class RAGPipeline:
    def __init__(self):
        pass
    def ingest(self,file_path,collection_name):
        ing=loader()
        text=ing.exportToMarkDownOutput(file_path)

        chunk=Chunking()
        parents, children = chunk.childToParentMetaDataEmbedding(text, file_path)

        store=Store()
        store.store_in_qdrant(parents, children,collection_name)
    
    def search(self, query,collection_name):
        search=HybridSearch()
        return search.retrieve_with_context(query,collection_name)



    # #step1: Extract
    # text = pipeline.export_to_markdown(file_path)
    # # step2: Process Hierarchical Chunks
    # parents, children = pipeline.childToParentMetaDataEmbedding(text, file_path)

    # pipeline.store_in_qdrant(parents, children)