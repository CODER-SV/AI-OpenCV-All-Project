import cv2,numpy,os
size = 4
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
datasets = 'datasets'
print('Training.....')
(images,labels,names,id) = ([],[],{},0)
for(subdirs,dirs,files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filenames in os.listdir(subjectpath):
            path = subjectpath + "/" + filenames
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id +=1
(width, height) = (130,100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)
cnt = 0
while True:
    _,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y+h,x:x+w]
        face_resizes = cv2.resize(face,(width, height))
        
        predictions = model.predict(face_resizes)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        if predictions[1] < 108:
            cv2.putText(img,'%s - %.0f' % (names[predictions[0]],predictions[1]),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
            print(names[predictions[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),3)
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("input.jpg",img)
                cnt = 0
    cv2.imshow('OpenCV',img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()