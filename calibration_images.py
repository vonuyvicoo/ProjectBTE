import cv2
import numpy as np

width = 640
height = 480



cap = cv2.VideoCapture(1) #left
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap2 = cv2.VideoCapture(3)
cap2.set(cv2.CAP_PROP_FPS, 30)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

num = 0

globalGamma = 2.5

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

while cap.isOpened():

    succes1, img = cap.read()
    succes2, img2 = cap2.read()

    img = adjust_gamma(img,gamma=globalGamma)
    img2 = adjust_gamma(img2,gamma=globalGamma)  

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(r'images/stereoLeft/imageL' + str(num) + '.png', img)
        cv2.imwrite(r'images/stereoright/imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)
