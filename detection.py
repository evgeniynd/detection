from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from threading import Timer
import os
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import urllib.request
import face_recognition

issend = 0
capbuff = []
motioncount = 0
KnownFacespath = 'KnownFaces'
images = []
classNames = []
myList = os.listdir(KnownFacespath)

# construct the argument parse and parse the arguments hostmjd
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rec", type=int, default=0,
                help="rec file")
ap.add_argument("-fps", "--fps", type=int, default=0,
                help="show fps in frame")
ap.add_argument("-f", "--face", type=int, default=0,
                help="face recognition")
ap.add_argument("-w", "--webhook", default='127.0.0.1',
                help="addres of webhook")
ap.add_argument("-n", "--name", default='noname',
                help="name of video stream")
ap.add_argument("-s", "--sourc", default=0,
                help="path to video stream")
ap.add_argument("-u", "--pause", type=int, default=100,
                help="pause recognize")
ap.add_argument("-p", "--prototxt", default='MobileNetSSD_deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default='MobileNetSSD_deploy.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

if(args['sourc'] == '0'): args['sourc'] = 0

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
absFilePath = os.path.abspath(__file__)
path, filename = os.path.split(absFilePath)
net = cv2.dnn.readNetFromCaffe(os.path.join(
    path, args["prototxt"]), os.path.join(path, args["model"]))

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

#vs = VideoStream(src='d:\PROGRAMING\Projects\real-time-object-detection\KnownFaces\Agafonov.jpg').start()
vs = VideoStream(src=args["sourc"]).start()
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

def setnull():
    global issend, motioncount
    issend = 0
    motioncount = 0

def recflie():   
    if args['rec'] == 0: capbuff.clear()
    now = datetime.now()
    dtString = now.strftime("%H%M%S") 
    if len(capbuff) >= 10:
               
        output = cv2.VideoWriter('Resources/video/'+dtString +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),20,(w,h))                              
        for i in capbuff:
            output.write(i)
        print('Записано Resources/video/output_video_from_file'+dtString +'.mp4')
        print("Записано " + str(len(capbuff)) + " кадров")

        link = "http://"+args["webhook"]+"/objects/?script=objectDetect&fobject=" + sss + "&sourc=" + \
                        args["name"] + '&time='+str(dtString) + '&filename='+dtString +'.mp4&detect=0'
        try:
            urllib.request.urlopen(link)
        except:
            print('Адрес не доступен ' + link)        
        output.release()
        capbuff.clear()
    else:
        link = "http://"+args["webhook"]+"/objects/?script=objectDetect&sourc=" + \
                            args["name"] + '&time='+str(dtString) + '&detect=0'
        try:
            urllib.request.urlopen(link)
        except:
            print('Адрес не доступен ' + link)
    

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def getflles():
    for cls in myList:
        curImg = cv2.imread(f'{KnownFacespath}/{cls}')
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])
    print(classNames)
    print(myList)

if args['face'] == 1:
    getflles()    
    encodeListKnown = findEncodings(images)
    print("Декодирование закончено")

def crop_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 19)
    for (x,y,w,h) in faces:
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

# loop over the frames from the video stream
while True:    
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1024)

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    myfps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    myfps = int(myfps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    #myfps = str(myfps)
 
    # putting the FPS count on the frame
    if args['fps']: cv2.putText(frame, str(myfps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
  
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction

        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # print(box.astype("int"))
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            
            font_path = 'Fonts/Roboto-Regular.ttf'
            font = ImageFont.truetype(font_path, 18)
            #color = ImageColor.getrgb(COLORS[idx])            

            sss = str(CLASSES[idx])
            # print("[INFO] {}".format(label))
            if sss == "person" or sss == "dog" or sss == "cat":

                # Распознаем лицо
                if sss == "person" and args['face'] == 1:

                    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                    facesCurFrame = face_recognition.face_locations(imgS)
                    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        # print(faceDis)
                        matchIndex = np.argmin(faceDis)                        

                        if matches[matchIndex]:
                            name = classNames[matchIndex]
                            print(name)
                            y1, x2, y2, x1 = faceLoc                            
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            #cv2.rectangle(frame, (x1, y2 - 35), (x2, y2),(0, 255, 0), cv2.FILLED)
                            img_pil = Image.fromarray(frame)
                            b, g, r, a = 0, 255, 0, 0
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((x1 + 6, y1 - 30), str(name),
                                font=font, fill=(b, g, r, a))
                            frame = np.array(img_pil) 
                              
                            now = datetime.now()
                            dtString = time.time()   
                            link = "http://"+args["webhook"]+"/objects/?script=objectDetect&person=" + name + "&sourc=" + args["name"] + '&time='+str(dtString)
                            try:
                                f = urllib.request.urlopen(link)
                            except:
                                print('Адрес не доступен ' + link)                       
                        else:
                            name = 'Не известное лицо'
                            print(name)
                            now = datetime.now()
                            dtString = now.strftime("%H%M%S")
                            filename = 'UnKnownFaces/face-'+dtString+'.jpg'                            
                            print("Лицо сохранено")
                            y1, x2, y2, x1 = faceLoc                            
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cropped = frame[y1:y2, x1:x2]                            
                            imggr = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                            cv2.imwrite(filename, imggr)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            #cv2.rectangle(frame, (x1, y2 - 35), (x2, y2),(0, 255, 0), cv2.FILLED)
                            img_pil = Image.fromarray(frame)
                            b, g, r, a = 0, 255, 0, 0
                            draw = ImageDraw.Draw(img_pil)           
                            draw.text((x1 + 6, y1 - 30), str(name),
                                font=font, fill=(b, g, r, a))
                            frame = np.array(img_pil)   

                motioncount = args['pause']

                if sss == "person":
                    label = "Человек: {:.2f}%".format(confidence * 100)
                if sss == "dog":
                    label = "Собака: {:.2f}%".format(confidence * 100)
                if sss == "cat":
                    label = "Кошка: {:.2f}%".format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 1)
                y = startY - 15 if startY - 15 > 15 else startY + 15                
                img_pil = Image.fromarray(frame)
                b, g, r, a = 0, 255, 0, 0
                draw = ImageDraw.Draw(img_pil)
                draw.text((startX + 6, startY - 30), str(label),
                          font=font, fill=(b, g, r, a))
                frame = np.array(img_pil)

                if issend == 0:
                    now = datetime.now()                    
                    dtString = now.strftime("%H%M%S")
                    issend = 1
                    filename = 'Resources/photo/'+sss + '-'+str(dtString)+'.jpg'
                    cv2.imwrite(filename, frame)
                    link = "http://"+args["webhook"]+"/objects/?script=objectDetect&fobject=" + sss + "&sourc=" + \
                        args["name"] + '&time='+str(dtString) + '&filename='+sss + '-'+str(dtString)+'.jpg&detect=1'
                    try:
                        f = urllib.request.urlopen(link)
                    except:
                        print('Адрес не доступен ' + link)                    
                    if args['rec'] == 1: Timer(60.0, setnull).start()   
                                                
    if(motioncount > 0):
        motioncount = motioncount - 1
    if(motioncount == 0):
        issend = 0  

    # show the output frame
    if motioncount > 0:
        if args['rec'] == 1: 
            capbuff.append(frame)
        else:
            if len(capbuff) < 1: capbuff.append(frame)
    else:
        if len(capbuff) != 0:            
            recflie()
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    # update the FPS counter    
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
