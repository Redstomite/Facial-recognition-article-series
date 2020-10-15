import cv2
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('dataset/trainer.yml')
haarCascadeClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
webCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
df = pd.read_csv("dataset/dataset.csv")

while True:
    ret, currentFrame = webCam.read()
    currentFrameInBW = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
    facesInCurrentFrame = haarCascadeClassifier.detectMultiScale(
        currentFrameInBW,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for (xCoord, yCoord, width, height) in facesInCurrentFrame:
        cv2.rectangle(currentFrame, (xCoord, yCoord), (xCoord+width, yCoord+height), (255, 0, 0), 2)
        id, confidence = recognizer.predict(currentFrameInBW[yCoord:yCoord+height, xCoord:xCoord+width])
        row = df[df["IDNum"] == id]
        name = df.iloc[[0], 1].to_string(index=False)
        cv2.putText(currentFrame, name, (xCoord + 5, yCoord - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('camera', currentFrame)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("done")
webCam.release()
cv2.destroyAllWindows()
