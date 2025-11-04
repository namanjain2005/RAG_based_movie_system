import time
import numpy as np
import os
import json
from collections import defaultdict



class semantic_search():
    def __init__(self,model_name = 'all-MiniLM-L6-v2'): 
        print("importing sentence transfromers", time.time())
        ### this should not be here just it is fast for testing otherwise it should be in top
        from sentence_transformers import SentenceTransformer
        print("import complete",time.time())

        self.model = SentenceTransformer(model_name) # use similarity which was model trained on
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generating_embedding(self,token:str):
        if not token or token.isspace():
            raise ValueError("query is empty")
        emb = self.model.encode([token]) # list -> list 
        # print(emb)
        # print(len(emb))
        # print(type(emb[0]))
        return emb[0] # this is because token is just one list only

    def build_embedding(self,documents):
        self.documents = documents
        doc_descs = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_descs.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_descs,show_progress_bar=True)
        np.save('./cache/movie_embeddings.npy',self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self,documents:list[dict]):
        if(os.path.exists('./cache/movie_embeddings.npy')):
            self.embeddings = np.load('./cache/movie_embeddings.npy')
            if len(self.embeddings) == len(documents):
                self.documents = documents
                for i, doc in enumerate(self.documents): 
                    # make it more consistent by using id or enumerate(using i) though documents same but I think using id is more consistent
                    # so may want to use id if can
                    self.document_map[i] = doc
                return self.embeddings
        else:
            return self.build_embedding(documents)
    
    def search(self,query:str,limit:int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        emb =self.generating_embedding(query)
        sim_scores = []
        for doc_id,doc in self.document_map.items():
            score = cosine_similarity(emb,self.embeddings[doc_id])
            sim_scores.append((score,doc))
        sim_scores.sort(key=lambda x: x[0], reverse=True)
        ans = defaultdict()
        for (score,doc) in sim_scores[:limit]:
            ans[doc['id']] = {"score":score , "id" : doc['id'] , "title" : doc['title']}
        return ans


def search(query:str):
    ss = semantic_search()
    with open('./data/course-rag-movies.json','r') as f:
        documents = json.load(f)
    ss.load_or_create_embeddings(documents['movies'])
    return ss.search(query)
        
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_query_text(query:str):
    ss = semantic_search()
    embedding = ss.generating_embedding(query)
    # print(f"Query: {query}")
    # print(f"First 5 dimensions: {embedding[:5]}")
    # print(f"Shape: {embedding.shape}")
        

def embed_text(text:str):
    ss = semantic_search()
    embedding = ss.generating_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss  = semantic_search()
    print(f"{ss.model}")
    print(f"maxLength{ss.model.max_seq_length}")
    return

def verify_embeddings():
    ss = semantic_search()
    with open('./data/course-rag-movies.json','r') as f:
        documents = json.load(f)
    embeddings = ss.load_or_create_embeddings(documents['movies'])
    # print(f"Number of docs:   {len(documents)}")
    # print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")   