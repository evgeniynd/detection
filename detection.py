# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
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
from PIL import ImageFont, ImageDraw, Image, ImageColor
import urllib.request
import http.server
import socketserver

PORT = 8008

handler = http.server.SimpleHTTPRequestHandler

#with socketserver.TCPServer(("", PORT), handler) as httpd:
  #  print("Server started at localhost:" + str(PORT))
  #  httpd.serve_forever()

issend = 0
motioncount = 0
countobj = 0
memobj = 0
capbuff = []

# construct the argument parse and parse the arguments hostmjd
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webhook", default='192.168.45.10',
                help="addres of webhook")
ap.add_argument("-n", "--name", default='noname',
                help="name of video stream")
ap.add_argument("-s", "--sourc", default=0,
                help="path to video stream")
ap.add_argument("-u", "--pause", default=30,
                help="pause recognize")
ap.add_argument("-p", "--prototxt", default='MobileNetSSD_deploy.prototxt.txt',
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default='MobileNetSSD_deploy.caffemodel',
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

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

#vs = VideoStream(src='rtsp://admin:er143900@192.168.0.248:554/ISAPI/Streaming/Channels/101').start()
vs = VideoStream(src=args["sourc"]).start()
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


def setnull():
    global issend, motioncount
    issend = 0
    motioncount = 0

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

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
    myfps = str(myfps)
 
    # putting the FPS count on the frame
    cv2.putText(frame, myfps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

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

            sss = str(CLASSES[idx])
            # print("[INFO] {}".format(label))
            if sss == "person" or sss == "dog" or sss == "cat":

                motioncount = 100

                if sss == "person":
                    label = "Человек: {:.2f}%".format(confidence * 100)
                if sss == "dog":
                    label = "Собака: {:.2f}%".format(confidence * 100)
                if sss == "cat":
                    label = "Кошка: {:.2f}%".format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                font_path = 'Fonts/Roboto-Regular.ttf'

                font = ImageFont.truetype(font_path, 18)
                #color = ImageColor.getrgb(COLORS[idx])
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
                    filename = 'Resources/photo/'+sss + '-'+dtString+'.jpg'
                    cv2.imwrite(filename, frame)
                    link = "http://"+args["webhook"]+"/objects/?script=objectDetect&fobject=" + sss + "&name=" + \
                        args["name"] + '&time='+dtString + '&filename=' + \
                        filename + '&count={}'.format(countobj)
                    f = urllib.request.urlopen(link)
                    Timer(60.0, setnull).start()           
    if(motioncount > 0):
        motioncount = motioncount - 1
    if(motioncount == 0):
        issend = 0            
    #print(motioncount)

    # show the output frame
    if motioncount > 0:
        capbuff.append(frame)
    else:
        if len(capbuff) != 0:
            frame_width = 800
            frame_height = 600
            frame_size = (frame_width, frame_height)
            now = datetime.now()
            dtString = now.strftime("%H%M%S")
            cv2.imshow(dtString, capbuff[5])            
            #output = cv2.VideoWriter('Resources/video/output_video_from_file'+dtString +'.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25, frame_size)      
            output = cv2.VideoWriter('Resources/video/output_video_from_file'+dtString +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),myfps,(w,h))                              
            for i in capbuff:
                output.write(i)
            output.release()
            capbuff.clear()

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
