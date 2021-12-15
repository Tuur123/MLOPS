from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class MovieList(BaseModel):
    movies: list


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/movies/")
async def create_movieList(movielist: MovieList):
    return movielist