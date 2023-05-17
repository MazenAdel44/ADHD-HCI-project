#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
from dollarpy import Point , Recognizer , Template 
import os
import numpy as np


# In[2]:


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions


# In[3]:


templates=[] #list of templates for $1 training


# In[4]:


def getPoints(videoURL,label):
    cap = cv2.VideoCapture(videoURL)#web cam =0 , else enter filename
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #List to hold Coordinates
        points = []
        nose = []
        left_eye_inner=[]
        left_eye=[]
        left_eye_outer=[]
        right_eye_inner=[]
        right_eye=[]
        right_eye_outer=[]
        left_ear=[]
        right_ear=[]
        mouth_left=[]
        mouth_right=[]
    
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            if ret==True:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                                 

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                       )
                
              
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    index = 0
                    newlist=[]
                    for lnd in pose:
                        if(index  in [0,1,2,3,4,5,6,7,8,9,10]):
                            newlist.append(lnd)
                        index+=1



                    # add points of face
                    nose.append(Point(newlist[0].x,newlist[0].y,1))
                    left_eye_inner.append(Point(newlist[1].x,newlist[1].y,2))
                    left_eye.append(Point(newlist[2].x,newlist[2].y,3))
                    left_eye_outer.append(Point(newlist[3].x,newlist[3].y,4))
                    right_eye_inner.append(Point(newlist[4].x,newlist[4].y,5))
                    right_eye.append(Point(newlist[5].x,newlist[5].y,6))
                    right_eye_outer.append(Point(newlist[6].x,newlist[6].y,7))
                    left_ear.append(Point(newlist[7].x,newlist[7].y,8))
                    right_ear.append(Point(newlist[8].x,newlist[8].y,9))
                    mouth_left.append(Point(newlist[9].x,newlist[9].y,10))
                    mouth_right.append(Point(newlist[10].x,newlist[10].y,11))

                   
                    




                except:
                    pass

                cv2.imshow(label, image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    points=nose+left_eye_inner+left_eye+left_eye_outer+right_eye_inner+right_eye+right_eye_outer+left_ear+right_ear+mouth_left+mouth_right

    print(label)
    
    return points


# In[5]:


vid = r'C:\Users\DELL\OneDrive\Desktop\lr.mp4'
points = getPoints(vid,"Look Right") 
tmpl_2 = Template('Look Right', points)
templates.append(tmpl_2)


# In[6]:


vid = r'C:\Users\DELL\OneDrive\Desktop\ll.mp4'
points = getPoints(vid,"Look Left") 
tmpl_2 = Template('Look Left', points)
templates.append(tmpl_2)


# In[146]:


def classify():
    recognizer = Recognizer(templates)
    cap = cv2.VideoCapture(0)#web cam =0 , else enter filename
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #List to hold Coordinates
        points = []
        nose = []
        left_eye_inner=[]
        left_eye=[]
        left_eye_outer=[]
        right_eye_inner=[]
        right_eye=[]
        right_eye_outer=[]
        left_ear=[]
        right_ear=[]
        mouth_left=[]
        mouth_right=[]
    
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor Feed
            if ret==True:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                                 

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                       )
                
              
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    index = 0
                    newlist=[]
                    for lnd in pose:
                        if(index  in [0,1,2,3,4,5,6,7,8,9,10]):
                            newlist.append(lnd)
                        index+=1



                    # add points of face
                    nose.append(Point(newlist[0].x,newlist[0].y,1))
                    left_eye_inner.append(Point(newlist[1].x,newlist[1].y,2))
                    left_eye.append(Point(newlist[2].x,newlist[2].y,3))
                    left_eye_outer.append(Point(newlist[3].x,newlist[3].y,4))
                    right_eye_inner.append(Point(newlist[4].x,newlist[4].y,5))
                    right_eye.append(Point(newlist[5].x,newlist[5].y,6))
                    right_eye_outer.append(Point(newlist[6].x,newlist[6].y,7))
                    left_ear.append(Point(newlist[7].x,newlist[7].y,8))
                    right_ear.append(Point(newlist[8].x,newlist[8].y,9))
                    mouth_left.append(Point(newlist[9].x,newlist[9].y,10))
                    mouth_right.append(Point(newlist[10].x,newlist[10].y,11))
                    points=nose+left_eye_inner+left_eye+left_eye_outer+right_eye_inner+right_eye+right_eye_outer+left_ear+right_ear+mouth_left+mouth_right
                    result1 = recognizer.recognize(points)
                    print(result1)
                   
                    




                except:
                    pass
                cv2.imshow("classification", image)
                
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# In[147]:


classify()


# In[ ]:




