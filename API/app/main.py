from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class MovieList(BaseModel):
    movies: list

modelEndpoint = 'https://www.w3schools.com/python/demopage.php'

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/movies/")
async def create_movieList(movielist: MovieList):

    # stuur request naar azure model endpoint
    response = requests.post(url, data = movielist)

    return response.text