import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(folder_path, target_frame_count=90, image_size=(128, 128), test_size=0.2, random_seed=42):
    labels = []
    data = []

    for label, subfolder in enumerate(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".avi"):
                    video_path = os.path.join(subfolder_path, filename)
                    cap = cv2.VideoCapture(video_path)
                    frames = []

                    while cap.isOpened() and len(frames) < target_frame_count:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        resized_frame = cv2.resize(frame, image_size)
                        frames.append(resized_frame)

                    # 프레임 수가 부족하면 패딩을 추가
                    while len(frames) < target_frame_count:
                        frames.append(np.zeros_like(frames[0]))

                    frames = np.array(frames)
                    # 여기서 frames를 사용하여 모델에 데이터로 사용할 수 있음

                    labels.append(label)
                    data.append(frames)

                    cap.release()

    # 데이터를 4D 텐서로 변환 (샘플 수, 프레임 수, 높이, 너비)
    data = np.array(data)

    # 데이터를 0~1 사이의 값으로 정규화
    data = data.astype('float32') / 255.0

    # 라벨을 원-핫 인코딩으로 변환
    labels = np.array(labels)
    num_classes = len(os.listdir(folder_path))
    labels = np.eye(num_classes)[labels]

    # 데이터를 훈련 세트와 테스트 세트로 분할
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size, random_state=random_seed)

    print("Train data shape:", train_data.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test data shape:", test_data.shape)
    print("Test labels shape:", test_labels.shape)

    return train_data, test_data, train_labels, test_labels

# 사용 예시
folder_path = r"C:\PlayData\sign_remaster\Sign_Language_Remaster\data"  # AVI 파일이 있는 폴더 경로 입력
train_data, test_data, train_labels, test_labels = load_and_preprocess_data(folder_path)

### 전처리된 파일 재생
# 비디오 재생을 위한 함수 정의
def play_video(frames):
    for frame in frames:
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

    cv2.destroyAllWindows()

# # train_data 중 첫 번째 비디오 재생
# for i in range(len(train_data)):
#     play_video(train_data[i])