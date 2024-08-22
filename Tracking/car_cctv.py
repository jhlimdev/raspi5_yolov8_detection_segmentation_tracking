import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('yolov8n.pt')
def XY_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        xy = [x, y] 
        print(xy)

cv2.namedWindow('Car_velocity')
cv2.setMouseCallback('Car_velocity', XY_coordinate)

cap = cv2.VideoCapture('./veh2.mp4')

my_file = open("./coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()

cy1 = 322
cy2 = 368
offset = 6

vh_down = {}
vh_up = {}
down_counter = []
up_counter = []
violation = []
violation_id = ''

while True:
    ret, frame = cap.read() 
    if not ret:
        break
    
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.cpu()  # cuda 설정되어있는 pc에서 돌릴경우
    px = pd.DataFrame(a).astype("float")
    list1 = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        if ('car' in c) or ('truck' in c):
            list1.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list1)
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)
        
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if down_counter.count(id) == 0:
                    down_counter.append(id)
                    distance = 10
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    if a_speed_kh > 15:
                        violation.insert(0, id)
                            
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)  
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed_time = time.time() - vh_up[id]
                if up_counter.count(id) == 0:
                    up_counter.append(id)
                    distance = 10
                    a_speed_ms1 = distance / elapsed_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                    if a_speed_kh1 > 30:
                        violation.insert(0, id)
    
    cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, '1line', (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, '2line', (181, 368), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    
    down_count = len(down_counter)
    cv2.putText(frame, 'going down : ' + str(down_count), (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    
    up_count = len(up_counter)
    cv2.putText(frame, 'going up : ' + str(up_count), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, '2470027_ljh', (500, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

    violation_id = ' '.join(map(str, violation))
    cv2.putText(frame, 'violation : ' + violation_id, (700, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow('Car_velocity', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
