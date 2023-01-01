import cv2 as cv
import mediapipe as mp
import time
import hand_tracker_module as htm
import os

wcam = 640
hcam = 480

previous_time = 0
current_time = 0
cap = cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

images_list = []
images = os.listdir('images')

for image_path in images:
    image = cv.imread('images/'+image_path)
    images_list.append(image)

h,w,c = images_list[0].shape

detector = htm.handDetector()
if not cap.isOpened():
    cap = cv.VideoCapture(1)
if not cap.isOpened():
    raise IOError('Cannot open the camera')

while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    my_list = detector.find_positions(frame, draw=False)
    if len(my_list)!=0:
        print(my_list)
        if my_list[8][2] < my_list[7][2] and my_list[12][2] < my_list[11][2] and abs(my_list[8][1]-my_list[12][1])>40:
            frame[0:h, 0:w] = images_list[0]
        elif abs(my_list[8][2]-my_list[4][2])<18 and my_list[16][2]<=my_list[12][2]:
            frame[0:h, 0:w] = images_list[1]
        elif my_list[7][2]>my_list[8][2] and abs(my_list[8][1]-my_list[6][1])<30 and my_list[10][2]-my_list[6][2]>10:
            frame[0:h, 0:w] = images_list[4]
        elif my_list[4][2]>my_list[2][2]:
            frame[0:h, 0:w] = images_list[3]
        elif my_list[5][2]>my_list[11][2] and my_list[5][2]>my_list[15][2] and my_list[5][2]>my_list[19][2] and my_list[4][2]>my_list[3][2]:
            frame[0:h, 0:w] = images_list[2]

    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    cv.putText(frame, 'FPS:{}'.format(str(int(fps))), (520,40), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,255),3)
        
    cv.imshow('Frame', frame)

    if cv.waitKey(2) & 0xFF == ord('q'):
        break
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()