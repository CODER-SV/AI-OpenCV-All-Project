import cv2

alg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" #accessed the model file
haar_cascade = cv2.CascadeClassifier(alg) #loading the model file

cam = cv2.VideoCapture(0)
while True:
    _,img = cam.read()
    gryImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    face = haar_cascade.detectMultiScale(gryImg,1.3,5)#get cordinates of face image

    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()