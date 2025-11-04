import argparse
import json
from utils.preprocess import preprocess
from cli.search_utils import build_command,search_command,tf_command,idf_command,tf_idf_command,bm25_idf_command,bm25_tf_command,bm25_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("buildinvidx",help="To build Inverted Index") 
    
    tf_parser = subparsers.add_parser("tf",help= "to get term fequency")
    tf_parser.add_argument("docid",type=int, help="document ID")
    tf_parser.add_argument("term",help="term which you want to get freq of")

    idf_parser = subparsers.add_parser("idf",help="find IDF score of term")
    idf_parser.add_argument("term" , help="term for idf score")

    tf_idf_parser = subparsers.add_parser("tfidf" , help="tf-idf score")
    tf_idf_parser.add_argument("docid",type=int,help="document ID")
    tf_idf_parser.add_argument("term",help="term for tfidf score")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="bm25-idf-score")
    bm25_idf_parser.add_argument("term",help="term for getting bm25idf-score")

    bm25_tf_parser = subparsers.add_parser("bm25tf",help= "to get term fequency")
    bm25_tf_parser.add_argument("docid",type=int, help="document ID")
    bm25_tf_parser.add_argument("term",help="term which you want to get bm25tf of")
    bm25_tf_parser.add_argument("--b",type=float,help="tunable bm25 b parameter")
    bm25_tf_parser.add_argument("--k1",type=float,help="tunable bm25 k1 parameter")

    bm25_search_parser = subparsers.add_parser("bm25search" , help="search movie acc to query using bm25")
    bm25_search_parser.add_argument("query",help="query for serching")
    bm25_search_parser.add_argument("--b",type=float,help="tunable bm25 b parameter")
    bm25_search_parser.add_argument("--k1",type=float,help="tunable bm25 k1 parameter")
    bm25_search_parser.add_argument("--limit",type=int,help="no of movies")


    args = parser.parse_args()
    data_path = "./data/course-rag-movies.json"
    match args.command:
        case "search":
            docs = search_command(args.query)
            for doc in docs:
                print(doc["id"],doc["title"])
        case "buildinvidx":
            build_command()
        case "tf":
            tf = tf_command(args.docid,args.term)
            print(" hi - ", tf)
        case "idf":
            idf_score = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
        case "tfidf":
            tf_idf_score = tf_idf_command(args.docid,args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.docid}': {tf_idf_score:.2f}")
        case "bm25idf":
            bm25_idf_score = bm25_idf_command(args.term)
            print( f"BM25 IDF score of '{args.term}': {bm25_idf_score:.2f}")
        case "bm25tf":
            bm25_tf_score = bm25_tf_command(args.docid,args.term) # we didinot use k1,b for now
            print(f"BM25 TF score of '{args.term}' in document '{args.docid}': {bm25_tf_score:.2f}")
        case "bm25search":
            movies = bm25_search(args.query)
            for movie in movies:
                print(movie)
        case _:
            parser.print_help()



# def search_movie(query,data_path) -> list:
#     query = preprocess(query)
#     with open(data_path,'r') as f:
#         data = json.load(f)
#     movies = data['movies']

#     query_movie = []
#     for movie in movies:
#         temp = preprocess(movie['title'])
#         if(len(set(query).intersection(set(temp))) > 0):
#             query_movie.append(movie['title'])

#     return query_movie 


    

if __name__ == "__main__":
    main()