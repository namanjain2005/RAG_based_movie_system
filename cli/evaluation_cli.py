import argparse
import json
from cli.hybrid_search_cli import rrf_search
import time

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    parser.add_argument("--k",type=int,default=60,help="tunable parameter")
    parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="query enhancement method")
    parser.add_argument("--rerank_method",type=str,choices=['individual','batch','cross_encoder'],help="reranking method")


    args = parser.parse_args()
    limit = args.limit
    k = args.k
    enhance = args.enhance
    rerank_method = args.rerank_method

    # run evaluation logic here
    with open('./data/golden_dataset.json','r') as f:
        golden_dataset = json.load(f) # type: ignore
    
    total_retrived = 0
    relevant_retrived = 0
    total_relevant = 0
    for test_case in golden_dataset['test_cases'] :
        right_docs = 0
        query = test_case['query']
        relevant_docs = test_case['relevant_docs']
        n_relevant_docs = len(relevant_docs)
        retrived_docs = rrf_search(query,k,limit,enhance,rerank_method)
        total_retrived += limit # len(retrived_docs)
        total_relevant+= n_relevant_docs

        for doc in retrived_docs:
            if doc['title'] in relevant_docs:
                relevant_retrived +=1
                right_docs += 1
        
        print(f"\nquery - {query} \n precision - {right_docs/limit}")
        print(f"recall - {right_docs/n_relevant_docs}")
        print(retrived_docs)
        print(relevant_docs)
        print("\n")
        time.sleep(60)

    
    print(f"final precision is  {relevant_retrived/total_retrived}")
    print(f"final recall is {relevant_retrived/total_relevant}")






if __name__ == "__main__":
    main()