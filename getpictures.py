import cv2
import time
import pandas as pd

webCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
df = pd.read_csv("dataset/dataset.csv")

haarCascadeClassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = df.shape[0]+1
name = input("Enter Name: ")
time.sleep(2)
count = 0

while True:
    ret, currentFrame = webCam.read()
    currentFrameInBW = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
    facesInCurrentFrame = haarCascadeClassifier.detectMultiScale(
        currentFrameInBW,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    for (xCoord, yCoord, width, height) in facesInCurrentFrame:
        cv2.rectangle(currentFrame, (xCoord, yCoord), (xCoord+width, yCoord+height), (255, 0, 0), 2)
        count += 1
        cv2.imwrite("dataset/"+"Pictures/"+"User." + str(face_id) + '.' + str(count) + ".jpg",
                    currentFrameInBW[yCoord:yCoord+height, xCoord:xCoord+width])
        cv2.imshow('CapturedFace', currentFrame)
        print(count, " pictures taken")
        time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if count == 30:
        break

print("Done with pictures")
webCam.release()
cv2.destroyAllWindows()

df.loc[face_id, "Name"] = name
df.loc[face_id, "IDNum"] = face_id
df.to_csv("dataset/dataset.csv", index=False)
