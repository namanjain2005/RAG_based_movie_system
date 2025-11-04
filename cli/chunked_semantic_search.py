from cli.semantic_search import semantic_search,cosine_similarity
from cli.search_utils import sem_chunk_text
import numpy as np
import os
import json
from collections import defaultdict

class chunkedSemanticSearch(semantic_search):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__(model_name) 
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self,documents:list[dict]):
        self.documents = documents
        all_chunks = []
        metadatas = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            desc:str = doc['description']
            if desc == '' or desc.isspace():
                continue
            chunks = sem_chunk_text(desc,4,1)
            total_chunks = len(chunks)
            for j,chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadatas.append({
                    "doc_id":doc['id'],
                    "chunk_id" : j,
                    "total_chunks":total_chunks
                })
        chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        self.chunk_embeddings = chunk_embeddings
        self.chunk_metadata = metadatas
        np.save('./cache/chunk_embeddings.npy',self.chunk_embeddings)
        with open(os.path.join("./cache", "chunk_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents:list[dict])->np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if (os.path.exists("./cache/chunk_embeddings.npy") and os.path.exists("./cache/chunk_metadata.json")):
            self.chunk_embeddings = np.load("./cache/chunk_embeddings.npy")
            with open("./cache/chunk_metadata.json",'r') as f:
                chunk_metadata = json.load(f)
            self.chunk_metadata  = chunk_metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def semantic_search(self,query:str,limit:int)->list[dict]:
        query_emb = self.generating_embedding(query)
        chunk_score = defaultdict(lambda:-1.0) # doc_id to chunk_score
        for i,chunk in enumerate(self.chunk_metadata): # type: ignore
            chunk_emb = self.chunk_embeddings[i] # type: ignore
            score = cosine_similarity(query_emb,chunk_emb)
            temp = chunk_score[chunk['doc_id']]
            chunk_score[chunk['doc_id']] = max(temp,score)
        
        chunk_score= sorted(chunk_score.items(), key=lambda item: item[1],reverse=True)[:limit]
        ans = []
        for doc_id,score in chunk_score:
            doc = self.document_map[doc_id]
            ans.append({
                'doc_id' : doc_id,
                'title' : doc['title'],
                # 'desc' : doc['description'][:100],
                'score':score
                
            })
        return ans
    
            
        