import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
from collections import Counter
from sentence_api import make_sentence
import json
import requests
import pickle

with open('G:/내 드라이브/LAB/Sign_Language_Remaster/logs/api_log.json',encoding='utf-8') as json_file:
    dic = json.load(json_file)
    dic = dic['Daily']
with open(r'G:\내 드라이브\LAB\Sign_Language_Remaster\logs\act_list.pkl', 'rb') as file:
    # 리스트 로드
    actions = pickle.load(file)
    print(len(actions),'개의 액션이 저장되어있습니다.')
seq_length = 30

# model = load_model(r"C:/PlayData/lstm_test100_9act_e50_C0_B0.h5")


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)


data = np.zeros((1, 156))
action_seq = []
this_action = '?'
class_select = []
word_list = []
CANT_FIND_HAND_COUNT = 0
while cap.isOpened():
    ret, img = cap.read()
    # img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        CANT_FIND_HAND_COUNT = 0
        da = []
        if len(result.multi_hand_landmarks) == 2 or len(result.multi_hand_landmarks) == 1:
            d= []
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 랜드마크 그려주기
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree
                #######################################
                angle = np.array([angle], dtype= np.float32)
                d.append(np.concatenate([joint.flatten(),angle.flatten()]))
                if len(result.multi_hand_landmarks)==1:
                    d.append(np.zeros_like(d[0]))
            da.append([np.concatenate(d)])
            if data.size != 0:
                data = np.vstack([data,np.concatenate(da)])
            else:
                data = da
        else:
            pass

        if len(data) < seq_length:
            continue

        input_data = np.expand_dims(np.array(data[-seq_length:], dtype=np.float32), axis=0)
        # api 호출
        array_list = input_data.tolist()
        url = 'http://203.250.133.192:8000/array'

        # POST 요청으로 바이너리 데이터 전송
        response = requests.post(url, json={"array": array_list})
        i_pred = response.json()['pred']
        # y_pred = model.predict(input_data, verbose=0).squeeze()
        # i_pred = int(np.argmax(y_pred))
        # top_classes = np.argsort(y_pred)[::-1][:1]
        # for i, class_idx in enumerate(top_classes):
        #     # print(f"상위 {i+1} 클래스: {class_idx}({actions[class_idx]}), 확률: {y_pred[class_idx]}")
        #     class_select.append(actions[class_idx])
        # print(11)
        # conf = y_pred[i_pred]

        # if conf < 0.8:
        #     continue
        print(len(actions),i_pred,type(i_pred))
        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action
        else:
            this_action = '?'
        cv2.putText(img, f'{this_action.upper()}',org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)
    else:
        CANT_FIND_HAND_COUNT+=1
        # 단어 탐색처리
        if class_select and CANT_FIND_HAND_COUNT>10:
            counter = Counter(class_select)
            print(counter)
            action_seq = [] #시퀀스 정리
            class_select=[]
            print(actions)
            #####################################################################
            if counter.most_common(1):

                most_common_element, count = counter.most_common(1)[0]
                word_list.append(most_common_element)
        
            
            
            
            
            #####################################################################
            
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) == ord('z'):
        # cv2.putText(img, f'Request sucess',org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)
        print(word_list)
        ans = make_sentence(word_list[1:])
        word_list = []
        print(ans)