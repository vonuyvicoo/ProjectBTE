import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import pygame
# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

model_path = '/Users/vonuyvico/Downloads/BTE-final/efficientdet_lite0.tflite'

globalGamma = 2.5
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
def normalizeC(coord_value, basis):
    return (float(basis)/2) - coord_value






pygame.mixer.init(size=32)



width2 = 640
height2 = 480

# Open both cameras
cap_right = cv2.VideoCapture(1)  
cap_right.set(cv2.CAP_PROP_FPS, 30)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width2)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height2)  

cap_left =  cv2.VideoCapture(4)
cap_left.set(cv2.CAP_PROP_FPS, 30)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width2)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height2)  

# Stereo vision setup parameters
frame_rate = 120   #Camera frame rate (maximum at 120 fps)
B = 5.5              #Distance between the cameras [cm]
f = 600              #Camera lense's focal length [mm]
alpha = 70       #Camera field of view in the horisontal plane [degrees]




# Main program loop with face detector and depth estimation using stereo vision


while(cap_right.isOpened() and cap_left.isOpened()):

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()

    frame_left = adjust_gamma(frame_left,gamma=globalGamma)
    frame_right = adjust_gamma(frame_right,gamma=globalGamma)  

################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

########################################################################################

    # If cannot catch any frame, break
    if not succes_right or not succes_left:                    
        break

    else:

        start = time.time()
        
        # Convert the BGR image to RGB
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)




        # Convert the RGB image to BGR
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


        frameG_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        frameG_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)


        #stereo = cv2.StereoBM_create(numDisparities=0, blockSize=5)
        #disparity = stereo.compute(frameG_left, frameG_right)
        #res = cv2.convertScaleAbs(disparity)
        #res = cv2.applyColorMap(res , cv2.COLORMAP_MAGMA)

        alpha2 = 0.5

        sum2 = cv2.addWeighted(frameG_left, alpha2, frameG_right, 1 - alpha2, 0.0)


        block_size = 11
        min_disp = 0
        max_disp = 256
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 50
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(frameG_left, frameG_right)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv2.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)
        disparity_SGBM2 = disparity_SGBM.copy()
        disparity_SGBM = cv2.applyColorMap(disparity_SGBM, cv2.COLORMAP_MAGMA)


    cv2.imshow("Disparity", disparity_SGBM)
        #cv2.imshow('pair', sum2)

        # Show the frames
        #cv2.imshow("frame right", frame_right) 
        #cv2.imshow("frame left", frame_left)


        # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


        ################## CALCULATING DEPTH #########################################################
'''
        center_right = 0
        center_left = 0

        if results_right.detections:
            for id, detection in enumerate(results_right.detections):

                bbox = detection.bounding_box

                h, w, c = frame_right.shape

                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                boundBox = bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                startR = start_point
                cv2.rectangle(frame_right, start_point, end_point, (255, 0, 0), 3)
                #cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


        if results_left.detections:
            for id, detection in enumerate(results_left.detections):

                bbox = detection.bounding_box

                h, w, c = frame_left.shape

                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                boundBox = bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

                center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                startL = start_point
                cv2.rectangle(frame_left, start_point, end_point, (255, 0, 0), 3)
                #cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


        # If no ball can be caught in one camera show text "TRACKING LOST"
        if not results_right.detections or not results_left.detections:
            cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

            cv2.putText(frame_right, "Distance: " + str(round(depth,1)), startR, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_left, "Distance: " + str(round(depth,1)), startL, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

            cv2.putText(frame_left, "+", (round(center_point_left[0]), round(center_point_left[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),3)
            cv2.putText(frame_right, "+", (round(center_point_right[0]), round(center_point_right[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),3)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            #print("Depth: ", str(round(depth,1)))

            print(center_point_left)
          

            buffer = np.sin(2 * np.pi * np.arange(44100) * 200 / 44100).astype(np.float32)
            sound = pygame.mixer.Sound(buffer)
            
            

            Lx,Ly = center_point_left
            panVector = normalizeC(Lx, width2) 
            panVector = panVector/320/2

            if(depth > 200):
                depth = 200

            indivDB = ((200-depth)/200) - 0.3
            sound.set_volume(indivDB)
            #print(indivDB)

            channel = pygame.mixer.find_channel()
            # pan volume full loudness on the left, and silent on right.

            channel.set_volume(0.5-panVector,0.5+panVector)
            #if(panVector > 0):
            #    channel.set_volume(0,panVector)
            #else:
            #    channel.set_volume(panVector*-1, 0)
            
            ################channel.play(sound)
            pygame.time.wait(int(sound.get_length() * 5))
            #print(str(center_point_left) + " | " + str(center_point_right))

'''

        

        #cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        #cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


        

        #cv2.imshow('disparity', res)
       

# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()