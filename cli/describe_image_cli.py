import argparse
import mimetypes
from utils.AG import rewrite_image_query

def main():
    parser = argparse.ArgumentParser(description = "image describing cli")
    subparser = parser.add_subparsers(dest="command",help="available command")

    describe_img_parser = subparser.add_parser("desc_img",help="to describe image")
    describe_img_parser.add_argument("img",help="path to an image")
    describe_img_parser.add_argument("query",help="query to rewrite based on image")

    args = parser.parse_args()

    match args.command:
        case "desc_img":
            mime,_ = mimetypes.guess_type(args.img)
            mime = mime or "image/jpeg"
            with open(args.img,'rb') as f:
                bin_image = f.read() 
            rewritten_query = rewrite_image_query(args.query,bin_image,mime)
            print(rewritten_query)
        case _ :
            parser.print_help()


if __name__ == "__main__":
    main()