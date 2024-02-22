import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os
#sasdf
# dataset 폴더 경로 설정
# dataset_folder = '/content/drive/MyDrive/LAB/Sign_Language_Remaster/code/lstm/dataset'

# dataset 폴더 아래의 모든 폴더 목록을 얻기
actions = ['(Blood) circulation', '(Facility) Bridge', '(Shooting gun)', '(Temperature)', '-jean', '-soup', 'a drawer', 'Acacia flower', 'Accomplice', 'airline', 'alcohol', 'All night', 'Anatomy', 'Anniversary', 'arithmetic', 'army unit', 'Assistant dog', 'attache', 'balance', 'barbershop', 'Be huge', 'Be insignificant', 'Be persistent', 'because', 'bone', 'breakthrough', 'bribe', 'buddhism', 'Bulguksa Temple', 'button', 'Celadon', 'Central office', 'chest', 'chewing gum', 'chicken', 'chinese character', 'church', 'Collection', 'Come across', 'Companion', 'confrontation', 'Construction site', 'copy machine', 'Crack (on the wall)', 'crumple', 'Defender', 'describe', 'Difficulty breathing', 'dog', 'dominance', 'Dressing table', 'Dual -ear', 'during', 'ear', 'earring', 'edit', 'egg plant', 'elder', 'engine', 'entrust', 'execution', 'expense', 'expensive', 'explanation', 'far', 'father', 'Federation', 'feel', 'Final exam', 'fire extinguisher', 'Five days', 'fix', 'Florist', 'flower', 'Football field', 'Gold', 'grasp', 'hair', 'Han River', 'Handling', 'hang', 'Hangul fingerprint', 'Hat (wearing)', 'Hawaii', 'hide', 'high heel', 'Historic sites', 'History', 'hold out', 'Hole', 'hot', 'House price', 'ignorance', 'Immature', 'Indifference', 'Insert', 'Insomnia', 'Installment', 'Irrelevant', 'Kalguksu', 'keep', 'Kim', 'knave', 'Korean Flag', 'law', 'Layer', 'Laziness', 'Lee Byung', 'length', 'let go', 'letter', 'lie', 'like', 'limp', 'long', 'lyrics', 'manicure', 'martyrdom', 'Mate', 'Material', 'meeting', 'Military uniform', 'miracle', 'model student', 'Money', 'Monthly', 'Moving', 'Multi -stage', 'My week', 'National examination', 'National treasure', 'nature', 'Navy', 'necessary', 'necktie', 'ninety', 'oblivion', 'Octopus', 'okay', 'One hundred', 'one room', 'only', 'organization', 'Outstream', 'Panama', 'Pass', 'persimmon', 'Photographer', 'pine nut', 'Pistol', 'Placebo', 'plan', 'Plaza', 'Pope', 'pot', 'Poverty', 'power plant', 'pregnancy', 'printing press', 'professional', 'Protection', 'Public', 'radish', 'rainbow', 'Rape', 'Reader', 'reading glasses', 'real', 'report', 'residence', 'road name', 'Rose of Sharon', 'rugby', 'Rule', 'safe', 'same age', 'sanity', 'school', 'science', 'secret', 'secretary', 'see', 'seizure', 'Seokdu', 'seoul', 'Seventh', 'seventy', 'sexual intercourse', 'Sgt', 'shave', 'shed', 'shelter', 'shoes', 'slaughter', 'Small trial', 'smock', 'Somehow', 'Songbyeolyeon', 'South Sea', 'spin', 'Spontaneous bullet', 'Stickiness', 'Stronger', 'struggle', 'swell', 'Taekwondo', 'Ten days', 'thailand', 'Thin', 'thorn', 'tie', 'To Wipe', 'together', 'tomato', 'train', 'Train station', 'tree', 'Troublesome', 'Turki Example Republic (abbreviated Turkiye)', 'Underneath', 'Unlimited', 'Ventilation', 'victim', 'vietnam', 'Village', 'vinyl', 'Visually impaired', 'walk', 'wayfarer', 'weeping', 'widow', 'wig', 'Writings', 'younger brother']

seq_length = 30

model = load_model(r"C:\PlayData\lstm_test31_234act_e100.h5")


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
        y_pred = model.predict(input_data).squeeze()
        i_pred = int(np.argmax(y_pred))
        top5_classes = np.argsort(y_pred)[::-1][:5]
        for i, class_idx in enumerate(top5_classes):
            print(f"상위 {i+1} 클래스: {class_idx}({actions[class_idx]}), 확률: {y_pred[class_idx]}")
        conf = y_pred[i_pred]

        if conf < 0.8:
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