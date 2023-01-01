import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.model_complexity = model_complexity
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
         self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):  
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                # for id, coors in enumerate(hand_landmarks.landmark):
                #     #print(id, coors)
                #     h, w, c = frame.shape
                #     cx, cy = int(coors.x*w), int(coors.y*h)
                #     print(id, cx, cy)
                #     if id == 0:
                #         cv.circle(frame, (cx,cy), 15, (255,0,255), cv.FILLED)
                    self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def find_positions(self, frame, hand_no=0, draw=True):
        my_list = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand_no]
            for id,hand_landmark in enumerate(myhand.landmark):
                h, w, c = frame.shape
                cx, cy = int(hand_landmark.x*w), int(hand_landmark.y*h)
                my_list.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 15, (255,0,255), cv.FILLED)

        return my_list
def main():
    previous_time = 0
    current_time = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()
    if not cap.isOpened():
        cap = cv.VideoCapture(1)
    if not cap.isOpened():
        raise IOError('Cannot open the camera')

    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        my_list = detector.find_positions(frame)
        if len(my_list)!=0:
            print(my_list)
        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time

        cv.putText(frame, str(int(fps)), (10,50), cv.FONT_HERSHEY_COMPLEX, 2, (255,0,255),3)
        
        cv.imshow('Frame', frame)

        if cv.waitKey(2) & 0xFF == ord('q'):
            break
        cv.waitKey(1)
    cap.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    main()