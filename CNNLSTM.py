import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# 設定參數
train_image_dir = './train_picture'
test_image_dir = './test_picture'
train_label_file = './train_labels.txt'
test_label_file = './test_labels.txt'

batch_size = 4  # LSTM 訓練時需要較小的 batch_size
time_steps = 50  # 每個影片有 50 幀
image_height, image_width = 144, 256
num_classes = 2  # 二分類


# 讀取標籤
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.read().strip().split('\n')
    label_list = []
    for line in labels:
        label_list.extend([int(x.strip()) for x in line.split(',')])
    return np.array(label_list, dtype=int)


# 影像序列生成器
def image_sequence_generator(image_dir, label_file, batch_size, time_steps, image_height, image_width):
    labels = load_labels(label_file)
    filenames = os.listdir(image_dir)
    num_samples = len(filenames) // time_steps  # 計算完整影片數量
    print(num_samples)
    while True:
        # 將資料分成一批一批，每批大小為 batch_size
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_labels = []

            # 計算該段影片的起始幀，並確認是否會超出圖片總數。
            for i in range(batch_size):
                start_idx = (offset + i) * time_steps
                if start_idx + time_steps > len(filenames):
                    break

                sequence_images = []
                sequence_labels = []

                # 處理 50 張圖片
                for j in range(time_steps):
                    image_path = os.path.join(image_dir, filenames[start_idx + j])
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"警告: 無法讀取圖片 {image_path}")
                        continue
                    image = cv2.resize(image, (image_width, image_height))
                    image = image.reshape((image_height, image_width, 1))
                    sequence_images.append(image)
                    sequence_labels.append(labels[start_idx + j])  # 收集 50 幀的標籤

                # 若成功收集 50 幀，才加入 batch
                if len(sequence_images) == time_steps:
                    batch_images.append(sequence_images)
                    
                    # 50 幀中 任意一幀標為 1（事故發生），整段影片就被標為 1。，否則為 0
                    video_label = 1 if sum(sequence_labels) > 0 else 0
                    batch_labels.append(video_label)

            # 轉成 numpy 格式並正規化 + One-hot 編碼
            batch_images = np.array(batch_images, dtype=np.float32) / 255.0
            batch_labels = to_categorical(batch_labels, num_classes=2)  # One-hot encoding

            yield batch_images, batch_labels

def video_level_labels(label_file):
    raw_labels = load_labels(label_file)
    video_labels = []
    for i in range(0, len(raw_labels), time_steps):
        clip_labels = raw_labels[i:i+time_steps]
        video_label = 1 if sum(clip_labels) > 0 else 0
        video_labels.append(video_label)
    return np.array(video_labels)



# 生成器
train_gen = image_sequence_generator(train_image_dir, train_label_file, batch_size, time_steps, image_height, image_width)
test_gen = image_sequence_generator(test_image_dir, test_label_file, batch_size, time_steps, image_height, image_width)

# 計算樣本數
num_train_samples = len(os.listdir(train_image_dir)) // time_steps
num_test_samples = len(os.listdir(test_image_dir)) // time_steps
print(f"number of train video: {num_train_samples}")
print(f"number of test video: {num_test_samples}")

# CNN 模型
def build_cnn():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(image_height, image_width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    return model


# 建立最終模型
cnn = build_cnn()

model = Sequential()
model.add(TimeDistributed(cnn, input_shape=(time_steps, image_height, image_width, 1)))  # 應用 CNN 到每一幀影像
model.add(LSTM(64, return_sequences=False))  # LSTM 負責學習時間序列資訊
model.add(Dense(2, activation='softmax'))  # 輸出層

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# 訓練模型
history = model.fit(train_gen, steps_per_epoch=num_train_samples // batch_size, epochs=10)

# 繪製 Loss 走勢圖
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 評估模型
score = model.evaluate(test_gen, steps=num_test_samples // batch_size, verbose=0)
print("test_loss:", score[0])
print("test_accuracy:", score[1])

# 預測測試集
y_true = video_level_labels(test_label_file)
y_pred_prob = model.predict(test_gen, steps=num_test_samples // batch_size)
y_pred = np.argmax(y_pred_prob, axis=1)  # 轉換為類別索引

# 計算混淆矩陣
cm = confusion_matrix(y_true[:len(y_pred)], y_pred)

# 顯示混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Crash', 'Crash'], yticklabels=['Non-Crash', 'Crash'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 印出 classification report
print(classification_report(y_true[:len(y_pred)], y_pred, target_names=['Non-Crash', 'Crash']))