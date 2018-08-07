import os
from os.path import join, isfile, isdir
import glob

import math
import numpy as np
import cv2
from scipy import interpolate

from TemplateTracker import TemplateTracker

class TraceGenerator(object):
    def __init__(self, img_dir_list, resize_rate=0.5):
        self.img_dir = img_dir_list
        self.resize_rate = resize_rate

    # windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    def smooth1D(self, x, window_len=11, window='flat'):
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len<3:
            return np.rint(x).astype(int)

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        # Reflected copies of the signal (with the window size) in both ends are introduced
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        y = np.rint(y[window_len/2:-(window_len/2):1]).astype(int)
        return y

    def interpolateTrace(self, pt_list, smooth_trace=True, show=False):
        window_type = 'flat'
        window_len = 11

        x = np.array([pt[0] for pt in pt_list])
        y = np.array([pt[1] for pt in pt_list])
        if smooth_trace:
            x = self.smooth1D(x, window_len, window_type)
            y = self.smooth1D(y, window_len, window_type)
        idx = np.arange(0, len(pt_list))

        trace_x = interpolate.splrep(idx, x, k=3)
        trace_y = interpolate.splrep(idx, y, k=3)

        idxnew = np.arange(0, len(pt_list)-0.5, 0.5)

        xnew = interpolate.splev(idxnew, trace_x, der=0)
        ynew = interpolate.splev(idxnew, trace_y, der=0)
        if smooth_trace:
            xnew = self.smooth1D(xnew, window_len, window_type)
            ynew = self.smooth1D(ynew, window_len, window_type)
        else:
            xnew = [int(round(xi)) for xi in xnew]
            ynew = [int(round(yi)) for yi in ynew]

        # img_idx = [int(math.floor(idx)) for idx in idxnew]
        img_idx = np.ceil(idxnew).astype(int)

        new_pt_list = zip( zip(xnew, ynew), img_idx )

        if show==True:
            bgr = cv2.resize( cv2.imread(self.img_dir[0], flags=1), (0,0), fx=self.resize_rate, fy=self.resize_rate)
            bgr2 = bgr.copy()

            pre_pt = new_pt_list[0][0]
            for pt, idx in new_pt_list[1:]:
                if idx%4==0:
                    bgr = cv2.line(bgr, pre_pt, pt, (0,0,255), 1)
                    pre_pt = pt
            cv2.imshow("Interpolated Trace", bgr)

            pre_pt = pt_list[0]
            for i, pt in enumerate(pt_list[1:]):
                if i%2==0:
                    # bgr2 = cv2.line(bgr2, pre_pt,pt, (0,255,0), 1)
                    bgr2 = cv2.circle(bgr2,pt,2,(255,0,0),-1)
                    pre_pt = pt
            cv2.imshow("Original Trace", bgr2)
                   
            k = cv2.waitKey(-1) & 0xff
            if k == 27:
                cv2.destroyAllWindows()

        return new_pt_list

if __name__ == "__main__":
    video_dir = './denseflow/MAH00399'
    img_dir = sorted(glob.glob(video_dir+'/img_*.jpg'))[0:79]
    obj_init_reg = [(371,290), (394,304)]
    obj_init_centroid = (382, 297)
    tracker = TemplateTracker(img_dir, 'region', init_reg=obj_init_reg)
    # tracker = TemplateTracker(img_dir, 'point', init_pt=obj_init_centroid)
    pt_list, reg_list, patch_list = tracker.track()

    traceGen = TraceGenerator(img_dir)
    traceGen.interpolateTrace(pt_list, show=True)