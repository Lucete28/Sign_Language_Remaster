# cd C:\Users\oem\Desktop\jhy\signlanguage\Sign_Language_Remaster\code\lstm; uvicorn FAST_API:app --reload --host 0.0.0.0
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
from concurrent.futures import ThreadPoolExecutor, as_completed
GROUP_SIZE = 25
MODELS = []
for i in range(GROUP_SIZE):
    print(f'{i}/{GROUP_SIZE}')
    model = load_model(f'C:/Users/oem/Desktop/jhy/signlanguage/Sign_Language_Remaster/model/2024-02-25_23-26-15/lstm_test103_G{i}_1645act_e20_C2_B0.h5')
    MODELS.append(model)
print('All models ready')
app = FastAPI()


@app.get("/")
def test():
    return {"message": "Hello World"}

class Item(BaseModel):
    array: list  # 넘파이 배열을 리스트로 받음



# async def predict_model(model, array):
#     start_time = time.time()
#     pred = model.predict(array, verbose=0).squeeze()
#     duration = time.time() - start_time
#     return np.argmax(pred), duration

re_li =[ [] for _ in range(GROUP_SIZE) ]

# @app.post("/receive")
# async def receive_array(request: Request):
#     # 데이터 받아서 변환
#     data = await request.json()
#     array_list = data['array']
#     array = np.array(array_list, dtype=np.float16)
#     for i, model in enumerate(MODELS):
#         pred = model.predict(array, verbose=0).squeeze()        
#         re_li[i].append(int(np.argmax(pred)))
#         #TODO conf 확인해서 처리 하도록(0.9이상?)
#     return {"status": "array received", "shape": array.shape, "CODE" : True}

#############################################################
def model_predict(model, array):
    pred = model.predict(array, verbose=0).squeeze()
    return int(np.argmax(pred))

@app.post("/receive")
async def receive_array(request: Request):
    data = await request.json()
    array_list = data['array']
    array = np.array(array_list, dtype=np.float16)
    
    # re_li = [[] for _ in range(GROUP_SIZE)]

    with ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(model_predict, model, array): i for i, model in enumerate(MODELS)}
        for future in as_completed(future_to_model):
            model_index = future_to_model[future]
            try:
                result = future.result()
            except Exception as exc:
                return {"CODE": False, "status": f'Model {model_index} generated an exception: {exc}'}
            else:
                re_li[model_index].append(result)
    print(re_li[0])
    return {"status": "array received", "shape": array.shape, "CODE": True, "tmp" : re_li[0]}
###############################################################




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
        return {"status": "Hello World","CODE":True, "pred_count" : final_confrim_li, "most_common_pred" : final_confrim_li[0][0], "most_common_count": final_confrim_li[0][1],"is_array_here":False}
    else:
        return {"status" : "NO DATA", "CODE":False}