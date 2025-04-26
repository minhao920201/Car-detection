import os
from sklearn.model_selection import train_test_split
import cv2
import random

def video_to_images(video_path, output_dir, start_idx, frames, num_images=50, kernel_size=(5, 5), sigma=0, resize_dim=(256, 144)):
    # Initialize the list to store images
    video_images = []
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return start_idx

    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) #50
    
    # Skip video if it does not have exactly 50 frames
    if total_frames != 50:
        print(f"Skipped: {video_path} has {total_frames} frames (expected 50)")
        video.release()
        return start_idx

    frames.append(total_frames)

    # Calculate the interval for frame extraction
    frame_interval = max(total_frames // num_images, 1)

    # Extract frames
    frame_idx = 0
    image_count = 0

    while image_count < num_images:
        ret, frame = video.read()  # 第一個值 ret 為 True 或 False，表示順利讀取或讀取錯誤，第二個值表示讀取到影片某一幀的畫面
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Apply Gaussian blur
            filtered_frame = cv2.GaussianBlur(frame, kernel_size, sigma)
            # Convert to grayscale
            gray_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
            # Resize the frame
            resized_frame = cv2.resize(gray_frame, resize_dim, interpolation=cv2.INTER_AREA)
            # Save the image
            image_path = os.path.join(output_dir, f"{start_idx:06d}.jpg")
            cv2.imwrite(image_path, resized_frame)
            video_images.append(resized_frame)

            start_idx += 1
            image_count += 1

        frame_idx += 1

    video.release()
    return start_idx



# Define directories
accident_dir = './videos/accident_videos'
normal_dir = './videos/normal_videos'
train_output_dir = './train_picture'
test_output_dir = './test_picture'
data_file = "./CCD_labels.txt"

# Create directories if not exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# Get video lists
accident_videos = [os.path.join(accident_dir, f) for f in os.listdir(accident_dir)]
normal_videos = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]

# Split into train and test sets
train_size = 0.7
accident_train, accident_test = train_test_split(accident_videos, train_size=train_size, random_state=42)
normal_train, normal_test = train_test_split(normal_videos, train_size=train_size, random_state=42)

# Combine accident and normal videos into train and test sets
train_videos = [(video, 1) for video in accident_train] + [(video, 0) for video in normal_train]
test_videos = [(video, 1) for video in accident_test] + [(video, 0) for video in normal_test]

# Randomly shuffle the videos
random.shuffle(train_videos)
random.shuffle(test_videos)

# Initialize label lists
train_labels = []
test_labels = []


# record number of frame in videos
train_frames = []
test_frames = []

# Process train videos
start_idx = 0
prev_idx = 0
for video, label in train_videos:
    start_idx = video_to_images(video, train_output_dir, start_idx, train_frames)
    if prev_idx == start_idx:
        continue
    train_labels.extend([label if label == 0 else int(os.path.basename(video).split('.')[0])])
    prev_idx = start_idx
print('transform train videos to images successfully')

# Process test videos
start_idx = 0
prev_idx = 0
for video, label in test_videos:
    start_idx = video_to_images(video, test_output_dir, start_idx, test_frames)
    if prev_idx == start_idx:
        continue
    test_labels.extend([label if label == 0 else int(os.path.basename(video).split('.')[0])])
    prev_idx = start_idx
print('transform test videos to images successfully')


# 讀取數據檔案
with open(data_file, 'r') as file:
    data_lines = [list(map(int, line.strip().split(", "))) for line in file.readlines()]

# get train labels
labels = []
idx = 0
for index in train_labels:
    if index == 0:
        labels.append([0] * train_frames[idx])  # 50 個 0
    else:
        labels.append(data_lines[index - 1])  # k-1 行的資料
    idx += 1


# Save train labels to file
with open('./train_labels.txt', 'w') as f:
    for sublist in labels:
        line = ', '.join(map(str, sublist))  # 将子列表转化为字符串
        f.write(line + '\n')            # 写入文件并换行
print('save train labels successfully')

# get test labels
labels = []
idx = 0
for index in test_labels:
    if index == 0:
        labels.append([0] * test_frames[idx])  # 50 個 0
    else:
        labels.append(data_lines[index - 1])  # k-1 行的資料
    idx += 1

# Save test labels to file
with open('./test_labels.txt', 'w') as f:
    for sublist in labels:
        line = ', '.join(map(str, sublist))  # 将子列表转化为字符串
        f.write(line + '\n')            # 写入文件并换行
print('save test labels successfully')