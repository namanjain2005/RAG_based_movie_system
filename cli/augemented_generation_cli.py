import argparse
from utils.AG import Augemented_generation,Summarization,citated_question_answers,question_answering

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--limit",type=int,default=5,help="Number of results")
    rag_parser.add_argument("--k",type=int,default=60,help="tunable parameter")
    rag_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="query enhancement method")
    rag_parser.add_argument("--rerank_method",type=str,choices=['individual','batch','cross_encoder'],help="reranking method")


    summarize_parser = subparsers.add_parser("summarize",help="summarize info regarding query")
    summarize_parser.add_argument("query",type=str,help="search query for summarizing")
    summarize_parser.add_argument("--limit",type=int,default=5,help="Number of results")
    summarize_parser.add_argument("--k",type=int,default=60,help="tunable parameter")
    summarize_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="query enhancement method")
    summarize_parser.add_argument("--rerank_method",type=str,choices=['individual','batch','cross_encoder'],help="reranking method")

    citated_question_answer_parser = subparsers.add_parser("citate",help="citate info regarding query")
    citated_question_answer_parser.add_argument("query",type=str,help="search query for summarizing")
    citated_question_answer_parser.add_argument("--limit",type=int,default=5,help="Number of results")
    citated_question_answer_parser.add_argument("--k",type=int,default=60,help="tunable parameter")
    citated_question_answer_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="query enhancement method")
    citated_question_answer_parser.add_argument("--rerank_method",type=str,choices=['individual','batch','cross_encoder'],help="reranking method")
    
    question_answer_parser = subparsers.add_parser("question",help="question/answers")
    question_answer_parser.add_argument("query",type=str,help="search query for summarizing")
    question_answer_parser.add_argument("--limit",type=int,default=5,help="Number of results")
    question_answer_parser.add_argument("--k",type=int,default=60,help="tunable parameter")
    question_answer_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="query enhancement method")
    question_answer_parser.add_argument("--rerank_method",type=str,choices=['individual','batch','cross_encoder'],help="reranking method")
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance
            rerank_method = args.rerank_method
            response = Augemented_generation(query,k,limit,enhance,rerank_method)
            print(response.text)
        case "summarize":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance
            rerank_method = args.rerank_method
            response = Summarization(query,k,limit,enhance,rerank_method)
            print(response.text)
        case "citate":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance
            rerank_method = args.rerank_method
            response = citated_question_answers(query,k,limit,enhance,rerank_method)
            print(response.text)
        case "question":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance
            rerank_method = args.rerank_method
            response = question_answering(query,k,limit,enhance,rerank_method)
            print(response.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()