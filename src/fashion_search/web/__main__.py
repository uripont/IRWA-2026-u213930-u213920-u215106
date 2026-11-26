"""Run the web app: uv run python -m fashion_search.web [--port 8088] [--debug]"""

import argparse

from fashion_search.web import create_app

parser = argparse.ArgumentParser(description="Run the Urban Outfinders web app.")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8088)  # the course template's port
parser.add_argument("--debug", action="store_true", help="reload on code changes")
args = parser.parse_args()

create_app().run(host=args.host, port=args.port, debug=args.debug)
