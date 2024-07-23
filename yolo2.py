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
import time

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
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils



pygame.mixer.init(size=32)



width2 = 640
height2 = 480

# Open both cameras
cap_right = cv2.VideoCapture(4)  
cap_right.set(cv2.CAP_PROP_FPS, 30)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width2)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height2)  
cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_left =  cv2.VideoCapture(1)
cap_left.set(cv2.CAP_PROP_FPS, 30)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width2)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height2)  
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Stereo vision setup parameters
frame_rate = 120   #Camera frame rate (maximum at 120 fps)
B = 5.5              #Distance between the cameras [cm]
f = 600              #Camera lense's focal length [mm]
alpha = 56       #Camera field of view in the horisontal plane [degrees]




# Main program loop with face detector and depth estimation using stereo vision
with mp_facedetector.FaceDetection(min_detection_confidence=0.4) as face_detection:

    while(cap_right.isOpened() and cap_left.isOpened()):

        succes_right, frame_right = cap_right.read()
        succes_left, frame_left = cap_left.read()

        

    ################## CALIBRATION #########################################################

        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    ########################################################################################

        # If cannot catch any frame, break
        if not succes_right or not succes_left:                    
            break

        else:
            frame_left = adjust_gamma(frame_left,gamma=globalGamma)
            frame_right = adjust_gamma(frame_right,gamma=globalGamma)  
            start = time.time()
            
            # Convert the BGR image to RGB
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results_right = face_detection.process(frame_right)
            results_left = face_detection.process(frame_left)

            # Convert the RGB image to BGR
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


            frameG_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            frameG_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)


            stereo = cv2.StereoBM_create(numDisparities=0, blockSize=5)
            disparity = stereo.compute(frameG_left, frameG_right)
            res = cv2.convertScaleAbs(disparity)
            res = cv2.applyColorMap(res , cv2.COLORMAP_MAGMA)

            ################## CALCULATING DEPTH #########################################################

            center_right = 0
            center_left = 0

            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_right.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)


            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)

                    bBox = detection.location_data.relative_bounding_box

                    h, w, c = frame_left.shape

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

                    cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)




            # If no ball can be caught in one camera show text "TRACKING LOST"
            if not results_right.detections or not results_left.detections:
                cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            else:
                # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
                # All formulas used to find depth is in video presentaion
                depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

                cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
                #print("Depth: ", str(round(depth,1)))

                cv2.putText(frame_left, "+", (int(center_point_left[0]), int(center_point_left[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                cv2.putText(frame_right, "+", (int(center_point_right[0]), int(center_point_right[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
              

                buffer = np.sin(2 * np.pi * np.arange(44100) * 200 / 44100).astype(np.float32)
                sound = pygame.mixer.Sound(buffer)
                
                

                Lx,Ly = center_point_left
                panVector = normalizeC(Lx, width2) 
                panVector = panVector/320/2 * -1

                if(depth > 200):
                    depth = 200

                indivDB = ((200-depth)/200) - 0.3
                sound.set_volume(indivDB)
                print(indivDB)

                channel = pygame.mixer.find_channel()
                # pan volume full loudness on the left, and silent on right.

                channel.set_volume(0.5-panVector,0.5+panVector)
                #if(panVector > 0):
                #    channel.set_volume(0,panVector)
                #else:
                #    channel.set_volume(panVector*-1, 0)
                
                channel.play(sound)
                pygame.time.wait(int(sound.get_length() * 5))
                #print(str(center_point_left) + " | " + str(center_point_right))



            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            #print("FPS: ", fps)

            cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


            

            cv2.imshow('disparity', res)


            # Show the frames
            cv2.imshow("frame right", frame_right) 
            cv2.imshow("frame left", frame_left)


            # Hit "q" to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()