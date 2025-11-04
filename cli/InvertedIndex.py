from collections import defaultdict,Counter
from utils.preprocess import preprocess
import json
import os
import pickle
import math

class InvertedIndex():
    def __init__(self)->None:
        self.index = defaultdict(set) # token -> docids 
        self.docmap = defaultdict() # id -> movie object
        self.term_freqs = defaultdict(Counter) # id -> counter dict
        self.doc_lengths = defaultdict() # id -> length
    
    def __add_document(self,docid:int,text:str)->None:
        tokens = preprocess(text)
        for token in tokens:
            self.index[token].add(docid)
        self.term_freqs[docid].update(tokens)
        self.doc_lengths[docid] = len(tokens)
        return

    def __get_avg_doc_length(self)->float:
        total_length_sum  = sum(self.doc_lengths.values())
        total_docs = len(self.doc_lengths)
        if total_docs == 0:
            return 0
        return total_length_sum/total_docs

    def get_tf(self,doc_id:int,term:str)->int:
        token = preprocess(term) # but note term is still "one" only ,may be lower would be enough
        if len(token)!=1:
            raise ValueError("there must be atleast one token")
        return self.term_freqs[doc_id][token[0]]
        
    def get_idf(self,term:str)->float:
        total_docs = len(self.docmap)
        terms = preprocess(term)
        if len(terms) != 1:
            raise ValueError("term must be a single token")
        term_doc_count = len(self.index[terms[0]])
        return math.log((total_docs + 1) / (term_doc_count + 1) )
    
    def get_bm25_idf(self,term:str)->float:
        tokens = preprocess(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        total_docs = len(self.docmap)
        term_doc_count = len(self.index[tokens[0]])
        return math.log((total_docs - term_doc_count + 0.5) / (term_doc_count + 0.5 ) + 1)
    
    def get_bm25_tf(self,docid:int,term:str,k1:float,b:float)->float:
        tf = self.get_tf(doc_id=docid,term=term)

        length_norm = 1 - b + b*((self.doc_lengths[docid]/self.__get_avg_doc_length()))
        return (tf*(k1 + 1))/(tf+k1*length_norm)
    
    def get_bm25(self,docid:int,term:str,k1:float,b:float)->float:
        bm25_tf = self.get_bm25_tf(docid,term,k1,b)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf*bm25_idf
    
    def bm25_search(self,query:str,k1:float,b:float,limit:int)->list:
        tokens = preprocess(query)
        doc_score = defaultdict(float)
        for token in tokens:
            for doc_id in self.index[token]:
                score = self.get_bm25(doc_id,token,k1,b)
                doc_score[doc_id] += score
        
        sorted_docs = (sorted(doc_score.items(),key=lambda x: x[1],reverse=True)[:limit])
        results = []
        for doc_id, score in sorted_docs:
            doc = self.docmap[doc_id]
            formatted_result = {
                "doc_id": doc["id"],
                "title": doc["title"],
                # "document": doc["description"],
                "score": score,
            }   
            results.append(formatted_result)

        return results

                

    
    def get_documents(self,term:str) -> list[int]:
        term = term.lower()
        docids = self.index.get(term,set()) # empty set if no term
        return sorted(list(docids))
    
    def build(self)->None:
        with open("./data/course-rag-movies.json",'r') as f: # may should not hard code it
            data = json.load(f)
        movies = data['movies']
        for movie in movies:
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[movie['id']] = movie
            self.__add_document(movie['id'],doc_description)
    
    def save(self)->None:
        os.makedirs("./cache", exist_ok=True)
        with open("./cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("./cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("./cache/term_freqs.pkl","wb") as f:
            pickle.dump(self.term_freqs,f)  
        with open("./cache/doc_lengths.pkl" , "wb") as f:
            pickle.dump(self.doc_lengths,f)

    def load(self)->None:
        try :
            with open("./cache/index.pkl","rb") as f:
                self.index = pickle.load(f)
            with open("./cache/docmap.pkl" , "rb") as f:
                self.docmap = pickle.load(f)
            with open("./cache/term_freqs.pkl" , "rb") as f:
                self.term_freqs = pickle.load(f)
            with open("./cache/doc_lengths.pkl" , "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError as e:
            print(f"file {e.filename} does not exist")
        return








