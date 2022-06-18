
""" Simple function to generate video from frames by Roberto Pellerito"""

import cv2
import os
import math

image_folder = '/home/rpellerito/trackformer/data/EXCAV/output/EXCAV/test'
video_name = '/home/rpellerito/trackformer/data/EXCAV/EXCAV_detected.avi'
images = []

list = sorted(os.listdir(image_folder))
for img in list:
    images.append(img)
    print(img)

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 
            fourcc = cv2.VideoWriter_fourcc(*'MP4V'), 
            fps=30, 
            frameSize=(width,height))
            

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
