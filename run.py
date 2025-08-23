# project_oracle/run.py
import uvicorn

if __name__ == "__main__":
    """
    This is the main entry point for running the Uvicorn server.
    
    Running this script as the main program (`python run.py`) ensures that
    the project's root directory is added to Python's path, which solves
    many common import errors.
    
    The string "app.main:app" tells Uvicorn:
    - Look for a file named `main.py` inside a package named `app`.
    - Inside that file, find the FastAPI instance variable named `app`.
    
    `reload=True` enables hot-reloading, so the server will automatically
    restart whenever you save a change to a code file.
    """
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )