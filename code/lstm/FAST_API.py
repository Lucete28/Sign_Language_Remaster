# uvicorn FAST_API:app --reload --host 0.0.0.0
# .\apienv\Scripts\activate
import sys
print("Python version")
print(sys.version)
try:
    import tensorflow as tf
    print("TensorFlow is installed")
    print(tf.__version__)
except ImportError:
    print("TensorFlow is not installed")


from tensorflow.keras.models import load_model
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np



MODELS = []
for i in range(25):
    print(i)
    model = load_model(f'C:/Users/oem/Desktop/jhy/signlanguage/Sign_Language_Remaster/model/2024-02-25_23-26-15/lstm_test103_G{i}_1645act_e20_C2_B0.h5')
    MODELS.append(model)
    
app = FastAPI()


@app.get("/")
def test():
    return {"message": "Hello World"}

class Item(BaseModel):
    array: list  # 넘파이 배열을 리스트로 받음


@app.post("/array")
async def receive_array(request: Request):
    # JSON 데이터를 비동기적으로 파싱
    data = await request.json()

    # JSON 객체에서 배열 데이터를 추출
    array_list = data['array']

    # 리스트를 넘파이 배열로 변환
    array = np.array(array_list, dtype=np.float16)
    pred = MODELS[0].predict(array, verbose=0).squeeze()
    # 배열 처리 (예: 출력)
    # print(array.shape)

    return {"status": "array received", "shape": array.shape, "array": array_list, "pred" : pred}