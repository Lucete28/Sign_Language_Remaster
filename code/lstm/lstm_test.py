import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# dataset 폴더 경로 설정
dataset_folder = '/content/drive/MyDrive/LAB/Sign_Language_Remaster/code/lstm/dataset'

# dataset 폴더 아래의 모든 폴더 목록을 얻기
actions = ['chewing gum', 'Fat', 'Final exam', 'hide', 'lie', 'Octopus', 'Stickiness', 'To Wipe', 'wayfarer', 'wig']

seq_length = 30

model = load_model(r"C:\PlayData\lstm_test21_10_e50.h5")


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

while cap.isOpened():
    ret, img = cap.read()
    # img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
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
            # data.append(np.concatenate(da))
            if data.size != 0:
                data = np.vstack([data,np.concatenate(da)])
            else:
                data = da
        else:
            pass

        if len(data) < seq_length:
            continue

        input_data = np.expand_dims(np.array(data[-seq_length:], dtype=np.float32), axis=0)
        # print('#####33',input_data.shape)
        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.6:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action
        else:
            this_action = '?'

        cv2.putText(img, f'{this_action.upper()}',org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break