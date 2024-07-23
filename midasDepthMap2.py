import cv2
import torch
import time
import numpy as np
import os
import sys
import mediapipe as mp
import utils
from scipy.interpolate import RectBivariateSpline
import math
import pygame


np.set_printoptions(threshold=np.inf)


#Converting Depth to distance
def depth_to_distance(depth_value,depth_scale):
  return 1.0/(depth_value*depth_scale)

alpha = 0.5
previous_depth = 0.0
globalGamma = 2.5

#Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth

def normalizeC(coord_value, basis):
    return (float(basis)/2) - coord_value


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Load a MiDas model for depth estimation
#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

#midas = torch.hub.load('/Users/vonuyvico/Downloads/midas_v21_small_256.pt')
midas = torch.hub.load("intel-isl/MiDaS", model_type)
#midas = torch.hub.load('/home/vonuyvico/BTE-final/midas','DPT_SwinV2_T_256', path='dpt_swin2_tiny_256.pt',force_reload=True,source='local', pretrained=True)


# Move model to GPU if available
device = torch.device("mps")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Open up the video capture from a webcam
cap = cv2.VideoCapture(0)
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

first = 0
#mps_device = torch.device("mps")


x_coords = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
y_coords = [72, 144, 216, 288, 360, 432, 504, 576, 648, 720]

address_tuple = [('0', '0'), ('0', '1'), ('0', '2'), ('0', '3'), ('0', '4'), ('0', '5'), ('0', '6'), ('0', '7'), ('0', '8'), ('0', '9'), ('1', '0'), ('1', '1'), ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('1', '6'), ('1', '7'), ('1', '8'), ('1', '9'), ('2', '0'), ('2', '1'), ('2', '2'), ('2', '3'), ('2', '4'), ('2', '5'), ('2', '6'), ('2', '7'), ('2', '8'), ('2', '9'), ('3', '0'), ('3', '1'), ('3', '2'), ('3', '3'), ('3', '4'), ('3', '5'), ('3', '6'), ('3', '7'), ('3', '8'), ('3', '9'), ('4', '0'), ('4', '1'), ('4', '2'), ('4', '3'), ('4', '4'), ('4', '5'), ('4', '6'), ('4', '7'), ('4', '8'), ('4', '9'), ('5', '0'), ('5', '1'), ('5', '2'), ('5', '3'), ('5', '4'), ('5', '5'), ('5', '6'), ('5', '7'), ('5', '8'), ('5', '9'), ('6', '0'), ('6', '1'), ('6', '2'), ('6', '3'), ('6', '4'), ('6', '5'), ('6', '6'), ('6', '7'), ('6', '8'), ('6', '9'), ('7', '0'), ('7', '1'), ('7', '2'), ('7', '3'), ('7', '4'), ('7', '5'), ('7', '6'), ('7', '7'), ('7', '8'), ('7', '9'), ('8', '0'), ('8', '1'), ('8', '2'), ('8', '3'), ('8', '4'), ('8', '5'), ('8', '6'), ('8', '7'), ('8', '8'), ('8', '9'), ('9', '0'), ('9', '1'), ('9', '2'), ('9', '3'), ('9', '4'), ('9', '5'), ('9', '6'), ('9', '7'), ('9', '8'), ('9', '9')]




pygame.mixer.init()
pygame.mixer.set_num_channels(100)
myvars = globals()

for add1, add2 in address_tuple:
    var_name = "sine_"+add1+"_"+add2
    file_name = "sine_"+add1+"_"+add2
    myvars[var_name] = pygame.mixer.Sound('/Users/vonuyvico/Downloads/BTE-final/output/'+file_name+'.wav')
    myvars[var_name].play(loops=-1)

    myvars[var_name].set_volume(0.05)








while cap.isOpened():

    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = adjust_gamma(img,gamma=globalGamma)  
    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()


    #prediction.to(mps_device)

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)


    #Creating a spline array of non-integer grid
    h , w = depth_map.shape
    #print(h,w)
    x_grid = np.arange(w)
    y_grid = np.arange(h)
        # Create a spline object using the output_norm array
    spline = RectBivariateSpline(y_grid, x_grid, depth_map)





    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    
    #x_coord = 400
    #y_coord = 10
    

    #depth_scale = 1
    #depth_mid_filt = spline(y_coord,x_coord) #Y AND X
    #depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
    #depth_mid_filt = (apply_ema_filter(depth_midas)/10)[0][0]


    #depthCalc = np.format_float_positional(depth_mid_filt , precision=3)



    ##POINT 1-----------------------------------------------------------------------------------
    
    tensorNum = 0
    for x_coordItem in x_coords:


        x_coord1 = x_coordItem - 64

        tensorNum2d = 0
        for y_coordItem in y_coords:
            y_coord1 = y_coordItem - 64

            depth_scale_1 = 1
            depth_mid_filt_1 = spline(y_coord1,x_coord1) #Y AND X
            depth_midas_1 = depth_to_distance(depth_mid_filt_1, depth_scale_1)
            depth_mid_filt_1 = (apply_ema_filter(depth_midas_1)/10)[0][0]    

            depthCalc_1 = np.format_float_positional(depth_mid_filt_1 , precision=3)


            depthCalcLimited = 0
            if(float(depthCalc_1) > 2):
                depthCalcLimited = 22
            else:
                depthCalcLimited = depthCalc_1



            depthCalcLimitedNorm = (1-(float(depthCalcLimited)))
            depthCalcLimitedNorm = depthCalcLimitedNorm * 0.3 #scale down by 0.6 to avoid limiting

            



            var_name = "sine_"+str(tensorNum)+"_"+str(tensorNum2d)
            myvars[var_name].set_volume(depthCalcLimitedNorm)

            #if(tensorNum == 9 and tensorNum2d == 5):
                #print("Opposing Volume is: "+str(sine_0_5.get_volume()))
                #print("Actual Volume is: " + str(sine_9_5.get_volume()))


            cv2.putText(img, str(round(depthCalcLimitedNorm, 3)), (x_coord1, y_coord1), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3)


            xNorm = normalizeC(x_coord1, width) * -1
            percentX = xNorm/(width/2)
            azimuth = int(percentX*90) #limit to 90 deg

            yNorm = normalizeC(y_coord1, height)
            percentY = yNorm/(height/2)
            elevation = int(percentY*90) #limit to 70 deg






            #print("Tensor #("+str(tensorNum)+", "+str(tensorNum2d)+"): "+ depthCalc_1+" | Azimuth: "+str(azimuth)+ " | Elevation: "+ str(elevation))

            #img = cv2.rectangle(img, (x_coord1, y_coord1), (x_coord1+25, y_coord1+30), (0,0,255), -1)

            tensorNum2d+=1
        tensorNum+=1

    print(fps)    
    

    #-------------------------------------------------------------------------------------------

    

    #xNorm = normalizeC(x_coord, width) * -1
    #percentX = xNorm/(width/2)
    #angleX = int(percentX*70) #limit to 70 deg
    #print(angleX)

    #yNorm = normalizeC(y_coord, height)
    #percentY = yNorm/(height/2)
    #angleY = int(percentY*90) #limit to 70 deg
    #print(angleY)





    #Displaying the distance.
    #cv2.putText(img, "Depth in unit: " + str(depthCalc), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #                2, (255, 255, 255), 3)






    # Create a named colour
    #red = [0,0,255]

    # Change one pixel
    


    imS = cv2.resize(img, (540, 304))
    imS = imS#adjust_gamma(imS,gamma=globalGamma)  
    dpS = cv2.resize(depth_map, (540, 304))  
    cv2.imshow('Image', imS)
    cv2.imshow('Depth Map', dpS)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cv2.destroyAllWindows()
cap.release()