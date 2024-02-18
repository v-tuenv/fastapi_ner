from typing import List

from pydantic import BaseModel


class WordOut(BaseModel):
    index: int
    text: str
    ner_tag: str
    start: int 
    end: int


class DocumentOut(BaseModel):
    tokens: List[WordOut] = []
