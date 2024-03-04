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
from collections import Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

GROUP_SIZE = 25
MODELS = []
for i in range(GROUP_SIZE):
    print(i)
    model = load_model(f'C:/Users/oem/Desktop/jhy/signlanguage/Sign_Language_Remaster/model/2024-02-25_23-26-15/lstm_test103_G{i}_1645act_e20_C2_B0.h5')
    MODELS.append(model)
print('All model ready')
app = FastAPI()


@app.get("/")
def test():
    return {"message": "Hello World"}

class Item(BaseModel):
    array: list  # 넘파이 배열을 리스트로 받음



async def predict_model(model, array):
    start_time = time.time()
    pred = model.predict(array, verbose=0).squeeze()
    duration = time.time() - start_time
    return np.argmax(pred), duration

re_li =[ [] for _ in range(GROUP_SIZE) ]

@app.post("/receive")
async def receive_array(request: Request):
    # 데이터 받아서 변환
    data = await request.json()
    array_list = data['array']
    array = np.array(array_list, dtype=np.float16)

    # start_time = time.time()  # 전체 예측 작업 시작 시간 측정

    # 각 모델에 대한 예측
    for i, model in enumerate(MODELS):
        pred = model.predict(array, verbose=0).squeeze()        
        re_li[i].append(int(np.argmax(pred)))
        #TODO conf 확인해서 처리 하도록(0.9이상?)

    # total_duration = time.time() - start_time  # 전체 예측 작업 완료 시간 측정


    # 가장 많이 예측된 클래스
    # number_counts = Counter(pred_list)
    # most_common_num, most_common_count = number_counts.most_common(1)[0]
    # most_common_num = int(most_common_num)  # 넘파이 int64를 파이썬 int로 변환

    # print(f"Total prediction time: {total_duration:.4f} seconds")  # 전체 예측 시간 출력

    return {"status": "array received", "shape": array.shape, "CODE" : True}


@app.get("/confirm")
def confirm():
    organize_li = []
    for re in re_li:
        
        if re: #리스트 비었을대 처리
            most_common_num, most_common_count = Counter(re).most_common(1)[0]
            organize_li.append(most_common_num)
    if organize_li:
        final_confrim_li = Counter(organize_li).most_common()

        for li in re_li:
            li.clear()
        return {"status": "Hello World","CODE":True, "pred_count" : final_confrim_li, "most_common_pred" : final_confrim_li[0][0], "most_common_count": final_confrim_li[0][1]}
    else:
        return {"status" : "NO DATA", "CODE":False}