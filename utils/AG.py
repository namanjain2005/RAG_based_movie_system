from cli.hybrid_search_cli import rrf_search
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def  Augemented_generation(query:str,k:int,limit:int,enhance:str = "None",rerank_method:str = "None"):

    retrieved_docs = rrf_search(query,k,limit,enhance,rerank_method)

    # print(retrieved_docs)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {retrieved_docs}

    Provide a comprehensive answer that addresses the query:"""

    
    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=(
                   prompt
                )
            )

    return  response


### conflicted resolution step needs to exist if the curated data is user created and not curated like ours
def Summarization(query:str,k:int,limit:int,enhance:str = "None",rerank_method:str = "None"):
    retrieved_docs = rrf_search(query,k,limit,enhance,rerank_method)
    
    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {retrieved_docs}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """
    
    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=(
                   prompt
                )
            )

    return  response

def citated_question_answers(query:str,k:int,limit:int,enhance:str="None",rerank_method:str="None"):
    retrieved_docs = rrf_search(query,k,limit,enhance,rerank_method)
    # print(retrieved_docs)
    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {retrieved_docs}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    # print(prompt)

    
    
    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=(
                   prompt
                )
            )
    
    # print(response)

    return  response


def question_answering(query:str,k:int,limit:int,enhance:str="None",rerank_method:str="None"):
    retrieved_docs = rrf_search(query,k,limit,enhance,rerank_method)
    
    prompt = f"""Answer the following question based on the provided documents.

    Question: {query}

    Documents:
    {retrieved_docs}

    General instructions:
    - Answer directly and concisely
    - Use only information from the documents
    - If the answer isn't in the documents, say "I don't have enough information"
    - Cite sources when possible

    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:"""

    # print(prompt)

    
    
    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=(
                   prompt
                )
            )
    
    # print(response)

    return  response


def rewrite_image_query(query:str,bin_image,mime:str):

    system_prompt = f"""
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    parts = [system_prompt,types.Part.from_bytes(data=bin_image, mime_type=mime),query.strip()] 

    response = client.models.generate_content(
        model='gemini-2.0-flash-001',
        contents=parts,
        )
    return response.text.strip() # type: ignore