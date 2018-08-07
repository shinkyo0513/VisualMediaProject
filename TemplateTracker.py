import os
from os.path import join, isfile, isdir
import glob

import numpy as np
import cv2

class TemplateTracker(object):
    def __init__(self, img_dir_list, init_mode, init_pt=(0,0), init_reg=[(0,0),(0,0)], resize_rate=0.5):
        self.init_mode = init_mode
        self.img_dir = img_dir_list
        self.init_pt = init_pt
        self.init_reg = init_reg
        self.resize_rate = resize_rate

    def shrinkToHighligh (self, roi_reg, bgr):
        roi_patch = bgr[roi_reg[0][1]:roi_reg[1][1], roi_reg[0][0]:roi_reg[1][0]]
        roi_gray = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2GRAY)
        _, roi_thres = cv2.threshold(roi_gray, 100, 255, 0)

        _, contours, _ = cv2.findContours(roi_thres, 1, 2)
        x,y,w,h = cv2.boundingRect(contours[0])
        x,y,w,h = x-int(round(0.15*w)), y-int(round(0.15*h)), int(round(1.3*w)), int(round(1.3*h))
        new_roi_reg = [(roi_reg[0][0]+x, roi_reg[0][1]+y), 
                            (roi_reg[0][0]+x+w, roi_reg[0][1]+y+h)]
        new_roi_patch = bgr[new_roi_reg[0][1]:new_roi_reg[1][1], new_roi_reg[0][0]:new_roi_reg[1][0]]
        return new_roi_reg, new_roi_patch

    def expandToHighlight (self, centroid, srch_reg, srch_patch):
        srch_gray = cv2.cvtColor(srch_patch, cv2.COLOR_BGR2GRAY)
        w,h = srch_gray.shape
        mask = np.uint8(np.zeros((w+2, h+2)))
        # loDiff: 100, upDiff: 60, Dc-loDiff <= D <= Dc+upDiff
        # low_diff = int(0.45*srch_gray)
        _, _, _, bbox = cv2.floodFill(srch_gray, mask, centroid, 255, 60, 60, cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

        x,y,w,h = bbox
        if x==0 or y==0:
            return None, None
        roi_patch = srch_patch[y:y+h, x:x+w]

        roi_reg = [(srch_reg[0][0]+x, srch_reg[0][1]+y), 
                        (srch_reg[0][0]+x+w, srch_reg[0][1]+y+h)]
        return roi_reg, roi_patch

    def track (self, show=True):
        pre_bgr = cv2.resize( cv2.imread(self.img_dir[0], flags=1), (0,0), fx=self.resize_rate, fy=self.resize_rate)

        if self.init_mode == 'region':
            pre_obj_reg = self.init_reg
            pre_obj_reg, pre_obj_patch = self.shrinkToHighligh(pre_obj_reg, pre_bgr)
            centroid = ( int(round((pre_obj_reg[1][0]+pre_obj_reg[0][0])/2)), int(round((pre_obj_reg[1][1]+pre_obj_reg[0][1])/2)) )
        elif self.init_mode == 'point':
            centroid = self.init_pt
            srch_reg = [(centroid[0]-40, centroid[1]-20), (centroid[0]+40, centroid[1]+20)]
            srch_patch = pre_bgr[srch_reg[0][1]:srch_reg[1][1], srch_reg[0][0]:srch_reg[1][0]]
            pre_obj_reg, pre_obj_patch = self.expandToHighlight((40, 20), srch_reg, srch_patch)
            assert pre_obj_reg is not None, 'The selected point is not on the light object!'
            centroid = ( int(round((pre_obj_reg[1][0]+pre_obj_reg[0][0])/2)), int(round((pre_obj_reg[1][1]+pre_obj_reg[0][1])/2)) )

        centroid_hist = [centroid, ]
        obj_reg_hist = [pre_obj_reg, ]
        obj_patch_hist = [pre_obj_patch, ]

        for img_idx in range(1, len(self.img_dir)):
            cur_bgr = cv2.resize( cv2.imread(self.img_dir[img_idx], flags=1), (0,0), fx=self.resize_rate, fy=self.resize_rate)
            cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

            pre_obj_reg_w = pre_obj_reg[1][0] - pre_obj_reg[0][0]
            pre_obj_reg_h = pre_obj_reg[1][1] - pre_obj_reg[0][1]
            w_ext = int(pre_obj_reg_w * 1)
            h_ext = int(pre_obj_reg_h * 1)
            l_move = d_move = 0
            if img_idx > 1:
                l_move += int((centroid_hist[-1][0]-centroid_hist[-2][0]))
                d_move += int((centroid_hist[-1][1]-centroid_hist[-2][1]))

            srch_reg = [(pre_obj_reg[0][0]-w_ext+l_move, pre_obj_reg[0][1]-h_ext+d_move), 
                            (pre_obj_reg[1][0]+w_ext+l_move, pre_obj_reg[1][1]+h_ext+d_move)]
            srch_patch = cur_bgr[srch_reg[0][1]:srch_reg[1][1], srch_reg[0][0]:srch_reg[1][0]]

            srch_res = cv2.matchTemplate(srch_patch, pre_obj_patch, 3)  #CV_TM_CCORR_NORMED
            cv2.normalize(srch_res, srch_res**3, 0, 1, cv2.NORM_MINMAX, -1)
            _, _, _, maxLoc = cv2.minMaxLoc(srch_res)

            centroid = ( maxLoc[0]+int(round(pre_obj_reg_w/2)), maxLoc[1]+int(round(pre_obj_reg_h/2)) )
            # cur_obj_reg, cur_obj_patch = shrinkToHighligh(cur_obj_reg, cur_bgr)
            cur_obj_reg, cur_obj_patch = self.expandToHighlight(centroid, srch_reg, srch_patch)
            if cur_obj_reg==None:   # when the tracked object disappeared
                break

            centroid = ( int(round((cur_obj_reg[1][0]+cur_obj_reg[0][0])/2)), int(round((cur_obj_reg[1][1]+cur_obj_reg[0][1])/2)) )
            # _, _, _, centroid = cv2.minMaxLoc(cv2.cvtColor(cur_obj_patch, cv2.COLOR_BGR2GRAY))
            # centroid = (cur_obj_reg[0][0]+centroid[0], cur_obj_reg[0][1]+centroid[1])

            centroid_hist.append(centroid)
            obj_reg_hist.append(cur_obj_reg)
            obj_patch_hist.append(cur_obj_patch)

            pre_obj_reg = cur_obj_reg
            pre_obj_patch = cur_obj_patch

            if show==True:
                cv2.rectangle(cur_bgr, srch_reg[0], srch_reg[1], (255, 0, 0), 1)
                cv2.rectangle(cur_bgr, cur_obj_reg[0], cur_obj_reg[1], (0, 255, 0), 1)
                cv2.imshow("image", cur_bgr)
                k = cv2.waitKey(-1) & 0xff
                if k == 27:
                    break
        cv2.destroyAllWindows()
        return centroid_hist, obj_reg_hist, obj_patch_hist

if __name__ == "__main__":
    video_dir = './denseflow/MAH00399'
    img_dir = sorted(glob.glob(video_dir+'/img_*.jpg'))
    obj_init_reg = [(371,290), (394,304)]
    obj_init_centroid = (382, 297)
    tracker = TemplateTracker(img_dir, 'region', init_reg=obj_init_reg)
    # tracker = TemplateTracker(img_dir, 'point', init_pt=obj_init_centroid)
    tracker.track()