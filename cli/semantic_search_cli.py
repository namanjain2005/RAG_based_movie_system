import argparse
import json
from cli.semantic_search import verify_model,embed_text,verify_embeddings,embed_query_text,search
from cli.search_utils import chunk_text,sem_chunk_text
from cli.chunked_semantic_search import chunkedSemanticSearch 

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command",help="available commands")

    subparser.add_parser("vf",help="to verify model")

    embed_text_parser = subparser.add_parser("embed",help="embed given text")
    embed_text_parser.add_argument("text",help= "text/query")

    subparser.add_parser("ver_emb",help="creating and checking embeddings")

    embed_query_parser = subparser.add_parser("query_emb",help = "embed given query")
    embed_query_parser.add_argument("query",help="query")

    semantic_search_parser = subparser.add_parser("search",help="semantic search")
    semantic_search_parser.add_argument("query",help="query for semantic search")
    semantic_search_parser.add_argument("--limit",type=int,help="no of queries")

    chunking_parser = subparser.add_parser("chunk",help="chunking the text")
    chunking_parser.add_argument("text",help="text to chunk")
    chunking_parser.add_argument("--chunk_size",default=200,type=int ,help="size of chunk")
    chunking_parser.add_argument("--overlap",default=0.0,type=float,help="overlap percent")

    semantic_chunking_parser = subparser.add_parser("semchunk",help="semantic chunking the text")
    semantic_chunking_parser.add_argument("text",help="text to semantic chunk")
    semantic_chunking_parser.add_argument("--chunk_size",default=4,type=int ,help="size of chunk")
    semantic_chunking_parser.add_argument("--overlap",default=0.0,type=int,help="no of sentences to overlap")

    chunked_search_parser = subparser.add_parser("chunk_search",help="semantic chunked search")
    chunked_search_parser.add_argument("query",help="query for searching")
    chunked_search_parser.add_argument("--limit",default=5,type=int)

    subparser.add_parser("build_ck_emb" , help="build chunk embeddings")

    args = parser.parse_args()

    match args.command:
        case "vf":
            verify_model()
        case "embed":
            embed_text(args.text)
        case "ver_emb":
            verify_embeddings()
        case "query_emb":
            embed_query_text(args.query)
        case "search":
            temp = search(args.query)
            print( "hi" , temp )
        case "chunk":
            chunks = chunk_text(args.text,args.chunk_size,args.overlap)
            print(chunks)
            print(len(chunks))
        case "semchunk":
            sem_chunks = sem_chunk_text(args.text,args.chunk_size,args.overlap)
            print(sem_chunks)
            print(len(sem_chunks))
        case "build_ck_emb":
            embed_chunks()
        case "chunk_search":
            chunk_search(args.query,args.limit)
        case _:
            parser.print_help()
        


    
def embed_chunks():
    with open("./data/course-rag-movies.json",'r') as f:
        documents = json.load(f)
    chunkSS = chunkedSemanticSearch()
    embeddings = chunkSS.load_or_create_chunk_embeddings(documents['movies'])
    print(f"Generated {len(embeddings)} chunked embeddings")
        
def chunk_search(query:str,limit:int):
    with open("./data/course-rag-movies.json",'r') as f:
        documents = json.load(f)
    chunkSS = chunkedSemanticSearch()
    embeddings = chunkSS.load_or_create_chunk_embeddings(documents['movies'])
    searchs = chunkSS.semantic_search(query,limit)
    print(searchs)
    return searchs

if __name__ == "__main__":
    main()