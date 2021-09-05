import cv2
import time
from imutils.video import FileVideoStream
import numpy as np
from skimage.io import imread_collection
from time import sleep

Net = cv2.dnn.readNetFromCaffe("SSDs/SSD_MobileNet_prototxt.txt" ,"SSDs/v2/MobileNetSSD_deploy.caffemodel")
video = FileVideoStream("【4K】Tokyo Walk - Ginza at Friday evening (July.2021) 【Japan】 (2).mp4").start()
image = ''
class_list = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",  "car", "cat",
              "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

colors = []
i = 0
input_shape = (320,320)

for i in range(len(class_list)):
    colors.append((np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))

col = imread_collection("Data/images/*.jpg")

prev_frame_time = 0
next_frame_time = 0

at = 0

while True:

    frame = video.read()
    # frame = col[at]
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    at += 1

    prev_frame_time = time.time()

    H,W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,input_shape),1/255, input_shape, 127.5)

    res = cv2.resize(frame , input_shape)
    # cv2.imshow("new" , frame)

    Net.setInput(blob)
    Detection = Net.forward()

    next_frame_time = time.time()

    fps = 1/(next_frame_time-prev_frame_time)

    fps = format(fps,".1f")

    i += 1

    cv2.putText(frame , f"{fps}" , (30,40) , cv2.FONT_HERSHEY_TRIPLEX , 0.8 , (0,255,255) , 1 )

    for object_index in np.arange(0,len(class_list)):

        confidence = Detection[0][0][object_index][2]
        box = Detection[0][0][object_index][3:7] * np.array([W,H,W,H])
        label = Detection[0][0][object_index][1]

        if confidence > 0.35:

            color = colors[int(label)]

            start_x , start_y , end_x , end_y = box.astype("int")

            cv2.rectangle(frame , (start_x,start_y) , (end_x,end_y) , color , 2)

            confidence = format(confidence,".2f")

            cv2.rectangle(frame,(start_x-5,start_y-15) , (start_x+110,start_y+5) , (255,255,255) , -1)

            cv2.putText(frame , f"{class_list[int(label)]} {confidence}" , (start_x,start_y) , cv2.FONT_HERSHEY_TRIPLEX , 0.5 , color , 1 )

    cv2.imshow("SSD" , frame)

    if cv2.waitKey(1)==ord(" ") :
        break


cv2.destroyAllWindows()


