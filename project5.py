import cv2, os
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
datasets = 'datasets'
sub_data = 'Shantanu' 

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)
(width, height) = (130,100)

face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

count = 1

while count<31:
    print(count)
    _,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        face_resizes = cv2.resize(face,(width, height))
        cv2.imwrite('%s/%s.png' % (path,count),face_resizes)
        count += 1
    cv2.imshow('OpenCV',img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
