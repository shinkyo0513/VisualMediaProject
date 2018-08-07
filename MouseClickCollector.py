import os
from os.path import join, isfile, isdir
import glob

import numpy as np
import cv2

class MouseClickCollector(object):
    def __init__(self, image):
        self.image = image
        self.backup = image.copy()
        self.refPt = []

    def getMouseClick(self, onlyOneClick=False):

        def click_and_crop(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.refPt = [(x, y)]
         
            elif event == cv2.EVENT_LBUTTONUP:
                self.refPt.append((x, y))

                cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0,255,0), 1)
                print self.refPt
                cv2.imshow("image", self.image)

        def only_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.refPt.append((x, y))

                cv2.circle(self.image, (x, y), 2, (0,255,0))
                cv2.imshow("image", self.image)

        while True:
            cv2.namedWindow("image")
            if onlyOneClick:
                cv2.setMouseCallback("image", only_click)
            else:
                cv2.setMouseCallback("image", click_and_crop)

            cv2.imshow("image", self.image)
            key = cv2.waitKey(-1) & 0xFF
            if key == ord("r"):
                self.image = self.backup.copy()
                self.refPt = []
            elif key == 27:
                break

        cv2.destroyAllWindows()
        return self.refPt

if __name__ == "__main__":
    video_dir = './denseflow/MAH00399'
    img_dir = sorted(glob.glob(video_dir+'/img_*.jpg'))
    image = cv2.resize( cv2.imread(img_dir[0], flags=1), (0,0), fx=0.5, fy=0.5)
    mouseReader = MouseClickCollector(image)
    mouseReader.getMouseClick(onlyOneClick=False)