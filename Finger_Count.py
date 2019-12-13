import cv2 as cv
import numpy as np
from sklearn.metrics import pairwise

background = None

accumulated = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 100
roi_left = 600

def count_fingers(thresholded,hand_segment):
    conv_hull = cv.convexHull(hand_segment)
    
    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    
    cX = (left[0] + right[0])/2
    cY = (top[1] + bottom[1])/2
    
    distance = pairwise.euclidean_distances([[cX,cY]],Y=[left,right,top,bottom])[0]
    max_distance = distance.max()
    radius = int(0.7*max_distance)
    circumference = (2*3.14*radius)
    
    circular_roi = np.zeros(thresholded.shape[:2],dtype = 'uint8')
    cv.circle(circular_roi,(int(cX),int(cY)),radius,255,10)
    circular_roi = cv.bitwise_and(thresholded,thresholded,mask = circular_roi)
    cv.imshow('Finger Contours',circular_roi)
    contours,hierarchy = cv.findContours(circular_roi.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    count = 0
    
    for cnt in contours:
        (x,y,w,h) = cv.boundingRect(cnt)
        
        out_wrist = (cY + (cY*0.25)) > (y+h)
        limit_points = ((circumference*0.25)>cnt.shape[0])
        
        if out_wrist and limit_points:
            count = count + 1
            
    return count
    
if __name__ == "__main__":
    count_fingers(thresholded,hand_segment)
