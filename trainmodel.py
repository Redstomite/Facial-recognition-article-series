import cv2
import os
import numpy as np
from PIL import Image



datasetPath = 'dataset/Pictures'
LBPH_Model = cv2.face.LBPHFaceRecognizer_create()
haarCascadeClassifer = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


all_faces = []
all_ids = []
all_image_paths = [os.path.join(datasetPath, img_path) for img_path in os.listdir(datasetPath)]
for imagePath in all_image_paths:
    image = Image.open(imagePath).convert('L')
    numpy_img = np.array(image, 'uint8')
    id = int(os.path.split(imagePath)[-1].split(".")[1])
    faces_in_img = haarCascadeClassifer.detectMultiScale(numpy_img)
    for (xCoord, yCoord, width, height) in faces_in_img:
        all_faces.append(numpy_img[yCoord:yCoord+height, xCoord:xCoord+width])
        all_ids.append(id)


LBPH_Model.train(all_faces, np.array(all_ids))

LBPH_Model.write('dataset/trainer.yml')
