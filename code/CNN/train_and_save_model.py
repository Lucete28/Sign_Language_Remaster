import tensorflow as tf
from tensorflow.keras import layers, models
def create_3d_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # 3D Convolution layers
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
# 모델 생성
input_shape = train_data.shape[1:]
num_classes = len(os.listdir(folder_path))
model = create_3d_cnn_model(input_shape, num_classes)

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_data, train_labels, epochs=3, batch_size=4, validation_split=0.2)

# 모델 평가
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')

# 모델을 저장할 경로 
model_save_path = 'C:/PlayData/sign_remaster/Sign_Language_Remaster/model/test10.h5'

# 모델 저장
model.save(model_save_path)

print(f'Model saved to {model_save_path}')
