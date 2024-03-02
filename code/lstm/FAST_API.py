# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()
app = FastAPI()

@app.get("/")
def test():
    return {"message": "Hello World"}

class Item(BaseModel):
    array: list  # 넘파이 배열을 리스트로 받음

@app.post("/array/")
def read_array(item: Item):
    # 리스트를 넘파이 배열로 변환
    np_array = np.array(item.array)
    # 배열 처리 수행 (예: 합계 계산)
    array_sum = np.sum(np_array)
    return {"sum": array_sum}