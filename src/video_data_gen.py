import cv2
import mediapipe as mp
import numpy as np
num = 10
cap = cv2.VideoCapture(0)  
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        continue

    # 손 감지 및 랜드마크 추출
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 검은색 배경 생성
    hand_landmarks_image = np.zeros_like(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크를 화면에 표시
            mp_drawing.draw_landmarks(hand_landmarks_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

    cv2.imshow('Video', hand_landmarks_image)

    # 녹화 시작ds
    if cv2.waitKey(1) & 0xFF == ord('s'):
        recording = True
        out = cv2.VideoWriter(f'C:/PlayData/sign_remaster/Sign_Language_Remaster/data/bread_house/bread_house_{num}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print("녹화 시작")
    # 녹화 중지 및 파일 저장
    if cv2.waitKey(1) & 0xFF == ord('t'):
        if recording:
            recording = False
            out.release()
            print(f"{num}번 영상 녹화 종료")
            num += 1    
            if num == 10:
                break
    # 녹화 중일 때 프레임 저장
    if recording:
        out.write(hand_landmarks_image)

    # 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()