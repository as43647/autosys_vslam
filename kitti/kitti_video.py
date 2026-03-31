import cv2
import os

image_folder = 'D:\\KITTI\\dataset\\dataset\\sequences\\03\\image_2'
video_name = 'kitti_03.mp4'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()