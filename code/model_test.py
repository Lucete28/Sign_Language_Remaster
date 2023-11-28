import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# 모델 파일 불러오기
model = load_model(r'Sign_Language_Remaster\model\test10.h5')

num = 0
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 88프레임을 담을 빈 배열 초기화
video_frames = [np.zeros((128, 128, 3), dtype=np.float32) for _ in range(88)]
frame_count = 0  # 현재까지 저장된 프레임 수

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

    # 이미지를 전처리하고 비디오 프레임 배열에 추가
    resized_image = cv2.resize(hand_landmarks_image, (128, 128))  # 입력 크기에 맞게 조정
    resized_image = resized_image / 255.0  # 정규화

    # 현재 프레임을 빈 배열 중 적절한 위치에 추가
    target_index = frame_count % 88
    video_frames[target_index] = resized_image
    frame_count += 1

    # 88프레임이 모였을 때 모델에 전달하여 예측
    if frame_count >= 88:
        input_video = np.array(video_frames)  # 88프레임을 4D 배열로 변환
        input_video = np.expand_dims(input_video, axis=0)  # 배치 차원 추가

        # 모델에 예측 요청
        pred = model.predict(input_video)  # 모델에 예측 요청

        # 예측된 클래스를 확인하고 라벨 출력
        predicted_class = np.argmax(pred, axis=1)  # 확률이 가장 높은 클래스의 인덱스 가져오기
        if predicted_class == 0:
            print("Predicted Label: house")
        else:
            print("Predicted Label: bread_house")

    cv2.imshow('Video', hand_landmarks_image)

    # 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
