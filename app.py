import torch
import numpy as np
import cv2
import time
import pickle

with open('CarParkPos', 'rb') as f: # for cctv1
    posList = pickle.load(f)

# with open('CarParkPos1', 'rb') as f: # for cctv2
#     posList = pickle.load(f)

width, height = 75, 75


class ObjectDetection:
    
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)


    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        model.classes = [2]
        return model


    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        labels, cord= results
        n = len(labels)
        center_point = []
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                centerx = ((x1 + x2) / 2)
                centery = ((y1 + y2) / 2)
                center_p = (int(centerx),int(centery))
                center_point.append(center_p)
                bgr = (0, 255, 0)
                cv2.circle(frame, center=center_p, radius=2, color=(0, 0, 255), thickness=2)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                # cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame, center_point



    def countParkingSpace(self, center_point, frame):
        spaceCounter = len(posList)
        for pos in posList:
            x, y = pos
            # spaceCounter = 0
            for i in range(len(center_point)):
                x1, y1 = center_point[i]

                if (x < x1 < (x + width)) and (y < y1 < (y + height)):
                   color = (255, 0, 0)
                   thickness = 7
                   spaceCounter -= 1
                else:
                    color = (0, 0, 255)
                    thickness = 2
                    # spaceCounter += 1
                cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, thickness)
        return spaceCounter
            

    def __call__(self):
        cap = cv2.VideoCapture('cctv1.mp4')
        index = 0
        while cap.isOpened():

            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame, center_point = self.plot_boxes(results, frame)
            #print(len(center_point))
            end_time = time.perf_counter()
            #print(index)
            counter = self.countParkingSpace(center_point, frame)
            index += 1
            
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.putText(frame, f'Parking Space: {int(counter)}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow("CCTV", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


detection = ObjectDetection()
detection()