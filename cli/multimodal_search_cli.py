import argparse
import json
from cli.lib.multimodal_search import MultimodalSearch

def main() -> None:
    parser = argparse.ArgumentParser(description = "image describing cli")
    subparser = parser.add_subparsers(dest="command",help="available command")

    search_using_image_parser = subparser.add_parser("search_with_image",help="to search movies using image")
    search_using_image_parser.add_argument("img",help="path to an image")

    args = parser.parse_args()

    match args.command:
        case "search_with_image":
            searches = image_search_command(args.img)
            print(searches)
        case _ :
            parser.print_help()


def image_search_command(img_path:str):
        with open('./data/course-rag-movies.json','r') as f:
            movies = json.load(f)
        movies = movies['movies']
        MMS = MultimodalSearch(movies)
        searches = MMS.search_with_image(img_path)
        return searches



if __name__ == "__main__":
    main()