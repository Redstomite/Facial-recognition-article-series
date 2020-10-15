import cv2

haarCascadeClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
webCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
    print(type(facesInCurrentFrame))
    for (xCoord, yCoord, width, height) in facesInCurrentFrame:
        cv2.rectangle(currentFrame, (xCoord, yCoord), (xCoord+width, yCoord+height), (255, 0, 0), 2)
        print(xCoord, yCoord, width, height
              )

    cv2.imshow('frame', currentFrame)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
