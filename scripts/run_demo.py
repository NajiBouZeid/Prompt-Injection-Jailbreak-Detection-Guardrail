import os

import uvicorn


def main():
    # Defaults to localhost for normal (non-Docker) use. The Docker image sets
    # HOST=0.0.0.0 so the server is reachable from outside the container via -p.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("demo.backend:app", host=host, port=port)


if __name__ == "__main__":
    main()
