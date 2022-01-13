import cv2
import imutils
import os
basedir = 'D:\\1_12\\frl2\\256'
dir = os.listdir(basedir)
for i, d in enumerate(dir):
    img = cv2.imread(os.path.join(basedir,d))
    img_rotated = imutils.rotate_bound(img, 90)
    cv2.imwrite(os.path.join(basedir,d),img_rotated)
