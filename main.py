import cv2

#Get user supplied values
cascPath = "../classifier/haarcascade_frontalface_default.xml"
cam.set(3, 1024)
cam.set(4, 576)

#Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    #Get the video from a capture device
    cam = cv2.VideoCapture(0)
    #Read the video into an image per frame
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    #Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)

    cv2.imshow("Camera", img)
    if cv2.waitKey(11) & 0xff == 27: #press ESC to quit program
        break

cam.release()
cv2.destroyAllWindows()