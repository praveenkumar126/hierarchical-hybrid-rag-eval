from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

import uuid
class Chunking:
    def __init__(self,model=None):
        if model is None:
            model="all-MiniLM-L6-v2"
        self.embedder=HuggingFaceEmbeddings(model_name=model)
    
    def semanticChunking(self,text:str, metadata:dict)->List[Document]:
        semantic_splitter = SemanticChunker(
            self.embedder, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90)
        doc=Document(page_content=text,metadata=metadata)
        return semantic_splitter.split_documents([doc])
    
    def recursiveCharacterBasedChunk(self,text,maxSize,seperators):
        if seperators is None:
            separators=["\n\n","\n","."," ",""]
        if len(text) <= maxSize:
            return [text]
        if not seperators:
            return [text[i:i+maxSize] for i in range(0, len(text), maxSize)]

        sep=seperators[0]
        if sep == "":
            return [text[i:i+maxSize] for i in range(0, len(text), maxSize)]
        
        current=""
        chunks=[]
        pages=text.split(sep)
        for page in pages:
            if current:
                candidate=current+sep+page
            else:
                candidate=page
            
            if len(candidate) <=maxSize:
                current=candidate
            else:
                if current:
                    chunks.extend(self.recursiveCharacterBasedChunk(current, maxSize, seperators[1:]))
                current=page
        chunks.extend(self.recursiveCharacterBasedChunk(current, maxSize, seperators[1:]))
        return chunks
        
    def childToParentMetaDataEmbedding(self, text, file_path):
        splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
        
        parentChunks = splitter.create_documents([text])
        #parentChunks = self.semanticChunking(text, {"source": file_path})
        childDocs = []
            
        for i, doc in enumerate(parentChunks):
            parentId = str(uuid.uuid4())
            doc.metadata["docId"] = parentId

                # --- NEW ARCHITECTURAL STEP: HEADER EXTRACTION ---
                # We look at the parent text to find the most relevant header
                # In Markdown, headers start with #. We grab the first one we find.
            parent_header = ""
            for line in doc.page_content.split('\n'):
                if line.strip().startswith('#'):
                    parent_header = line.replace('#', '').strip()
                    break
                # --------------------------------------------------

                # Use the specialized table-aware separators we discussed
            childText = self.recursiveCharacterBasedChunk(doc.page_content, 1000, ["\n\n", "\n", "|", " "])

            for c in childText:
                    # --- INJECT CONTEXT HERE ---
                    # We prepend the header so the Vector 'sees' the category
                contextual_text = f"Section: {parent_header} \nContent: {c}"
                    
                cDoc = Document(
                    page_content=contextual_text, 
                    metadata={
                        "parentId": parentId, 
                        "source": file_path, 
                        "isChild": True,
                        "section_header": parent_header, # Also keep it in metadata for filtering
                        "raw_text": c
                    }
                )
                childDocs.append(cDoc)
                    
        return parentChunks, childDocs


