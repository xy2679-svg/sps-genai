from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()

nlp = spacy.load("en_core_web_lg")


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = nlp(request.word)
    embedding = word.vector.tolist()

    return {
        "word": request.word,
        "embedding": embedding
      }