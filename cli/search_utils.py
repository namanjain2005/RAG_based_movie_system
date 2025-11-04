from cli.InvertedIndex import InvertedIndex
from utils.preprocess import preprocess
import re
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

# Tunable parameters
BM25_K1 = 1.5
BM25_B = 0.75


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query:str)->list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = preprocess(query)
    seen, results = set(), []
    for token in query_tokens:
        docids = idx.get_documents(token)
        for docid in docids:
            if docid in seen:
                continue
            
            seen.add(docid)
            results.append(idx.docmap[docid])
    return results

def tf_command(docid:int,term:str)->int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(docid,term)

def idf_command(term:str)->float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)
    

def tf_idf_command(docid:int,term:str)->float:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id=docid,term=term)
    idf = idx.get_idf(term=term)
    return tf*idf

def bm25_idf_command(term:str)->float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(docid:int,term:str,k1:float=BM25_K1,b1:float = BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(docid,term,k1,b1)

def bm25_search(query:str,k1:float = BM25_K1,b1:float = BM25_B,limit:int = 5):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query,k1,b1,limit)

def chunk_text(text:str, chunk_size:int,overlap:float)->list[str]:
    if overlap > 1 or overlap <=0:
        overlap = 1
    else:
        overlap = 1 - overlap
    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), int(chunk_size*overlap))
    ]
    return chunks

import re

def sem_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Step 1: Strip leading/trailing whitespace
    text = text.strip()

    # Step 2: Handle empty or whitespace-only input
    if not text:
        return []

    # Step 3: Split sentences by punctuation, but only if meaningful
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Step 4: If only one sentence and it doesn't end with punctuation, treat whole text as one
    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
        sentences = [text]

    # Step 5: Clean each sentence (remove extra spaces)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Step 6: Handle empty results after stripping
    if not sentences:
        return []

    # Step 7: Validate overlap/chunk_size logic
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("--overlap must be smaller than --max-chunk-size")

    # Step 8: Build chunks with overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks



def enhance_query(query:str,enhance:str)->str:
    if(enhance == "spell"):
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=(
                f"""Fix any spelling errors in this movie search query. 
                Only correct obvious typos. Don't change correctly spelled words.
                Query: "{query}"
                If no errors, return the original query.
                Corrected:"""
            )
        )
        print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
        query = response.text  # type: ignore
    elif(enhance == "rewrite"):
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=(
                f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:"""
            )
        )
        print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
        query = response.text  # type: ignore
    elif(enhance == "expand"):
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=(
                f"""Expand this movie search query with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                This will be appended to the original query.

                Examples:

                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                Query: "{query}"
                """ 
            )
        )
        print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
        query = response.text  # type: ignore
    return query
