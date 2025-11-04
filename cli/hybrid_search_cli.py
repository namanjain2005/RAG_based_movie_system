import argparse
import json
from cli.hybrid_search import HybridSearch
import os

from cli.search_utils import enhance_query



def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="Available commands")

    noramlize_parser = sub_parser.add_parser("normalize",help= "Normalization")
    noramlize_parser.add_argument("nums",nargs='+',type=float)

    weighted_search_parser = sub_parser.add_parser("weighted_search",help="to do a hybrid weighted search")
    weighted_search_parser.add_argument("query",help="query to search for")
    weighted_search_parser.add_argument("--alpha",type=float,default=0.5,help="weight of keyword_search")
    weighted_search_parser.add_argument("--limit",default=5,type=int,help="amount of search results")

    rrf_search_parser = sub_parser.add_parser("rrf_search",help="to do a rrf search")
    rrf_search_parser.add_argument("query",help="query to search for")
    rrf_search_parser.add_argument("--k",type=int,default=60,help="tunable parameter")
    rrf_search_parser.add_argument("--limit",default=5,type=int,help="amount of search results")
    rrf_search_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="query enhancement method")
    rrf_search_parser.add_argument("--rerank_method",type=str,choices=['individual','batch','cross_encoder'],help="reranking method")

    #,nargs='*',or giving it some default value

    args = parser.parse_args()

    match args.command:
        case "normalize":
           nums =  normalize(args.nums)
           print(nums)
        case "weighted_search":
            ws = weighted_search(args.query,args.alpha,args.limit)
            for s in ws:
                print(s)
        case "rrf_search":
            rrfs = rrf_search(args.query,args.k,args.limit,args.enhance,args.rerank_method)
            for rrf in rrfs: 
                print(rrf)
        case _:
            parser.print_help()


def normalize(nums: list[float])->list[float]:
    '''in place normalization'''
    if not nums:
        return nums

    min_score = min(nums)
    max_score = max(nums)

    if min_score == max_score:
        return [1.0]*len(nums)
    for i,num in enumerate(nums):
        nums[i] = (num-min_score)/(max_score-min_score)
    return nums

def weighted_search(query:str,alpha:float,limit:int):
    with open('./data/course-rag-movies.json','r') as f:
        document = json.load(f)
    document = document['movies']
    hs = HybridSearch(documents=document)
    return hs.weighted_search(query,alpha,limit)

def rrf_search(query:str,k:int,limit:int,enhance:str = "None",rerank_method= "None")->list[dict]:
    with open('./data/course-rag-movies.json','r') as f:
        document = json.load(f)
    document = document['movies']
    hs = HybridSearch(documents=document)

    query = enhance_query(query,enhance)

    searches = hs.rrf_search(query,k,limit*5)
    if rerank_method == "individual":
        searches = hs.individual_rerank(query,searches)
    elif rerank_method == "batch":
        searches = hs.batch_reranking(query,searches)
    elif rerank_method == "cross_encoder":
        searches = hs.cross_encoder_reranking(query,searches)
    
    return searches[:limit]

if __name__ == "__main__":
    main()