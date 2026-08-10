import uvicorn


def main():
    uvicorn.run("demo.backend:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
