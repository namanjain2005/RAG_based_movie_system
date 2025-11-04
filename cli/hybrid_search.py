import os
from dotenv import load_dotenv
from google import genai
import time
import json
from utils.preprocess import safe_json_loads
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")



from cli.InvertedIndex import InvertedIndex
from cli.chunked_semantic_search import chunkedSemanticSearch

BM25_K1 = 1.5
BM25_B = 0.75

class HybridSearch:
    def __init__(self, documents):
        self.doc_map = {doc["id"]: doc for doc in documents}
        self.semantic_search = chunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("./cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query:str,k1,b, limit:int):
        self.idx.load()
        return self.idx.bm25_search(query,k1,b,limit)
    
    def _chunked_semantic_search(self,query:str,limit:int):
        return self.semantic_search.semantic_search(query,limit)
    
    def _normalize(self,results):
        # Get all score values
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero if all scores are equal
        if min_score == max_score:
            for r in results:
                r["score"] = 1.0
        else:
            for r in results:
                r["score"] = (r["score"] - min_score) / (max_score - min_score)
        return results



    def weighted_search(self, query, alpha, limit=5):
        bm_25_searches = self._bm25_search(query,BM25_K1,BM25_B,limit*500)
        chunked_sem_searches = self._chunked_semantic_search(query,limit*500)
        normalized_bm_25_scores = self._normalize(bm_25_searches)
        normalized_chunked_sem_searches = self._normalize(chunked_sem_searches)
        # print(bm_25_searches,'\n' ,normalized_bm_25_scores) # i hypothize that both will be same
        
        bm_25_scores = {d["doc_id"] : d["score"] for d in normalized_bm_25_scores}
        chunked_sem_scores = {d["doc_id"] : d["score"] for d in normalized_chunked_sem_searches}

        combined = {}
        all_ids = set(bm_25_scores.keys()) | set(chunked_sem_scores.keys())
        for id in all_ids:
            bm_25_score = bm_25_scores.get(id,0.0)
            chunked_sem_score = chunked_sem_scores.get(id,0.0)
            weighted_score = alpha*bm_25_score + (1-alpha)*chunked_sem_score 
            doc = self.doc_map.get(id)
            if doc is None :
                print("could not find" , id , "in self.documents")
                continue
            combined[id] = {
                "doc_id":id,
                "title":doc["title"], # type: ignore
                "bm25_score":bm_25_score,
                "semantic_score":chunked_sem_score,
                "weighted_hybrid_score":weighted_score
            }
        sorted_docs = sorted(combined.values(), key=lambda x: x["weighted_hybrid_score"], reverse=True)[:limit]

        return sorted_docs
        


        
    ### can try where you have different k for keyword and semantic 
    def rrf_search(self, query, k:int, limit=10)->list[dict]:

        max_limit = 500*limit
        bm_25_searches = self._bm25_search(query,BM25_K1,BM25_B,max_limit)
        chunked_sem_searches = self._chunked_semantic_search(query,max_limit)
        bm_25_ranks = {doc['doc_id'] : i+1 for i,doc in enumerate(bm_25_searches)}
        sem_ranks = {doc['doc_id'] : i+1 for i,doc in enumerate(chunked_sem_searches)}
        combined = {}
        all_ids = set(bm_25_ranks.keys()) | set(sem_ranks.keys())

        for id in all_ids:
            bm_25_rank = bm_25_ranks.get(id)
            sem_rank = sem_ranks.get(id)

            rrf_score = 0
            if bm_25_rank is not None:
                rrf_score += (1/(k+bm_25_rank))
            if sem_rank is not None:
                rrf_score += (1/(k+sem_rank))

            doc = self.doc_map.get(id)
            if doc is None :
                print("could not find" , id , "in self.documents")
            
            combined[id] = {
                "doc_id":id,
                "title":doc["title"],#type:ignore
                "desc" : doc["description"],#type:ignore
                "bm_25_rank":bm_25_rank,
                "sem_rank":sem_rank,
                "rrf_score" :rrf_score
            }
        
        return sorted(combined.values(),key=lambda x:x['rrf_score'],reverse=True)[:limit]
    
    def individual_rerank(self,query:str,results:list[dict])->list[dict]:
        print("individual reranking")
        for result in results:
            client = genai.Client(api_key=api_key)
            
            doc = self.doc_map[result["doc_id"]]

            response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=(
                   f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {doc.get("title", "")} - {doc.get("description", "")}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Give me ONLY the number in your response, no other text or explanation.

                    Score:"""
                )
            )
            llm_score = int(response.text) # type: ignore
            result["llm_score"] = llm_score
            time.sleep(5)
        return sorted(results,key=lambda x:x["llm_score"],reverse=True)

    def batch_reranking(self,query:str,searches:list[dict])->list[dict]:
        print("batch reranking")
        lines = []
        for search in searches:
            desc:str = self.doc_map[search["doc_id"]]["description"]
            # print(desc)
            clean_desc = desc.replace("\n", " ").strip() 
            line = f"doc_id - { search['doc_id']} , title - {search['title']} ,description:{desc}"
            lines.append(line)
        doc_list_str = "\n".join(lines)
        # print(doc_list_str)
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=(
                   f"""Rank these movies by relevance to the search query.

                    Query: "{query}"

                    Movies:
                    {doc_list_str}

                    Return ONLY the IDs in order of relevance (best match first). Return ONLY string that can be desearlized by "json.loads" method. just string having list and nothing else . nothing else. For example:

                    [75, 12, 34, 2, 1]
                    """
            )
        )
        print(response.text)
        results = safe_json_loads(response.text)
        reranked_results = []
        for result in results:
            doc = self.doc_map[result]
            reranked_results.append({
                "doc_id" : doc['id'],
                "title" : doc['title'],
                # "desc" : doc['description']
            })
        return reranked_results
    def cross_encoder_reranking(self,query:str,searches:list[dict])->list[dict]:
        from sentence_transformers import CrossEncoder #i think it makes sense to load it lazily
        print("cross_encoder reranking")
        doc_scores = {}
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        for search in searches:
            doc_id = search['doc_id']
            doc = self.doc_map[doc_id]
            pair = [query,f"{doc['title']} - {doc['description']}"]
            score = cross_encoder.predict(pair)
            doc_scores[doc_id] = {
                "doc_id":doc_id,
                "title":doc['title'],
                # "desc":doc['description'],
                "score":score
            }
        return sorted(doc_scores.values(),key= lambda x:x['score'],reverse=True)







