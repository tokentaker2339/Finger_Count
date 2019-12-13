import cv2 as cv
import numpy as np
from sklearn.metrics import pairwise
from Finger_Count import *

background = None

accumulated = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 100
roi_left = 600

def calc_accum_avg(frame,accumulated_weight):
    
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv.accumulateWeighted(frame,background,accumulated_weight)
    
def segment(frame, threshold = 25):
    diff = cv.absdiff(background.astype('uint8'),frame)
    ret,thresholded = cv.threshold(diff,threshold,255,cv.THRESH_BINARY)
    #image,contours,hierarchy = cv.findContours(thresholded.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv.findContours(thresholded.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    else:
        #Largest Contour is the hand
        hand_segment = max(contours,key = cv.contourArea)
        
        return (thresholded, hand_segment)
    
def main():   
    cam =cv.VideoCapture(0)

    num_frames = 0

    while True:

        ret,frame = cam.read()
        frame_copy = frame.copy()
        roi = frame[roi_top:roi_bottom,roi_right:roi_left]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (7,7), 0)

        if num_frames < 60 :
            calc_accum_avg(gray,accumulated)

            if num_frames <= 59:
                cv.putText(frame_copy,'Wait Getting Background',(200,300),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv.imshow('Finger Count',frame_copy)

        else:
            hand = segment(gray)
            cv.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),2)

            if hand is not None:
                thresholded, hand_segment = hand

                #Draw Contours around Real hand in live stream
                cv.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5) 
                fingers = count_fingers(thresholded,hand_segment)
                cv.putText(frame_copy,str(fingers),(70,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv.imshow('Thresholded',thresholded)
                cv.imshow('Contours',frame_copy)

            elif hand is None:
                cv.imshow('Thresholded',np.zeros(shape = (roi_bottom-roi_top,roi_left-roi_right)))
                cv.imshow('Contours',frame_copy)

        num_frames = num_frames + 1

        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__": 
    main()


