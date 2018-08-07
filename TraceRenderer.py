import os
from os.path import join, isfile, isdir
import glob

import time
import numpy as np
import cv2

from TemplateTracker import TemplateTracker
from TraceGenerator import TraceGenerator
from MouseClickCollector import MouseClickCollector

class TraceRenderer(object):
    def __init__(self, img_dir_list, resize_rate=0.5):
        self.img_dir = img_dir_list
        self.resize_rate = resize_rate

    def multi_render(self, obj_list, write=False):
        if write:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('398_3.mp4',fourcc, 30.0, (960,540))

        init_bgr = cv2.resize( cv2.imread(self.img_dir[0], flags=1), (0,0), fx=self.resize_rate, fy=self.resize_rate)
        bgr_h, bgr_w = init_bgr.shape[0:2]
        frame_num = len(self.img_dir)
        max_pt_num = 0
        obj_num = len(obj_list)

        diff_thres = 0
        hl_thres = 120

        cur_mask = np.zeros_like(init_bgr)
        trace_mask = np.zeros_like(init_bgr)
        bgr_with_mask = np.zeros_like(init_bgr)

        for obj in obj_list:
            max_pt_num = max(max_pt_num, len(obj['new_pt_list']))

        pre_img_idx = -1
        for i in range(0, max_pt_num):
            cur_mask.fill(np.uint8(0))
            for obj in obj_list:
                if i < len(obj['new_pt_list']):
                    (cur_x,cur_y), cur_img_idx = obj['new_pt_list'][i]
                    ref_x, ref_y = obj['pt_list'][cur_img_idx]
                    delta_x, delta_y = cur_x-ref_x, cur_y-ref_y

                    cur_obj_reg = obj['reg_list'][cur_img_idx]
                    cur_obj_reg = [(cur_obj_reg[0][0]+delta_x, cur_obj_reg[0][1]+delta_y), (cur_obj_reg[1][0]+delta_x, cur_obj_reg[1][1]+delta_y)]
                    if cur_obj_reg[1][0]>bgr_w or cur_obj_reg[1][1]>bgr_h:
                        continue
                    cur_obj_patch = obj['patch_list'][cur_img_idx]
                    cur_mask[cur_obj_reg[0][1]:cur_obj_reg[1][1], cur_obj_reg[0][0]:cur_obj_reg[1][0]] = cur_obj_patch

            _, cur_mask_gray_hl = cv2.threshold(cv2.cvtColor(cur_mask, cv2.COLOR_BGR2GRAY), hl_thres, 0, cv2.THRESH_TOZERO)
            trace_mask_gray = cv2.cvtColor(trace_mask, cv2.COLOR_BGR2GRAY)
            diff = np.int32(cur_mask_gray_hl)-np.int32(trace_mask_gray)

            trace_mask = np.uint8(0.99*trace_mask)
            # trace_mask = np.where(np.expand_dims(cv2.cvtColor(trace_mask, cv2.COLOR_BGR2GRAY)>70, axis=3), trace_mask, np.uint8(0))

            # trace_mask = np.where(np.expand_dims(np.logical_and(diff>diff_thres, trace_mask_gray==0), axis=3), np.uint8(1.2*cur_mask), np.uint8(trace_mask))
            trace_mask = np.where(np.expand_dims(diff>diff_thres, axis=3), cur_mask, trace_mask)
            blur_trace_mask = trace_mask.copy()
            # blur_trace_mask = cv2.bilateralFilter(trace_mask, 5, 70, 70)
            # kernel_dilate = np.ones((3,3), np.uint8)
            # blur_trace_mask = cv2.dilate(blur_trace_mask, kernel_dilate, iterations = 1)
            blur_trace_mask = cv2.boxFilter(blur_trace_mask, -1, (3, 3), normalize=True)
            kernel_erode = np.ones((3,3), np.uint8)
            blur_trace_mask = cv2.erode(blur_trace_mask, kernel_erode, iterations = 1)

            if cur_img_idx != pre_img_idx:
                bgr = cv2.resize( cv2.imread(self.img_dir[cur_img_idx], flags=1), (0,0), fx=self.resize_rate, fy=self.resize_rate )
                bgr_with_mask = np.where(np.expand_dims(cv2.cvtColor(blur_trace_mask, cv2.COLOR_BGR2GRAY)>cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 
                                                                            axis=3), np.uint8(blur_trace_mask), bgr)
                cv2.imshow('mask', bgr_with_mask)
                if write:
                    out.write(bgr_with_mask)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cv2.imshow('mask', bgr_with_mask)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()

    # obj_list: a list of {'pt_list':pt_list, 'new_pt_list':new_pt_list, 'reg_list':reg_list, 'patch_list':patch_list}
    def multi_render_drawline(self, obj_list, write=False):
        if write:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('398_2.mp4',fourcc, 30.0, (960,540))

        init_bgr = cv2.resize( cv2.imread(self.img_dir[0], flags=1), (0,0), fx=self.resize_rate, fy=self.resize_rate)
        bgr_h, bgr_w = init_bgr.shape[0:2]
        frame_num = len(self.img_dir)
        max_pt_num = 0
        obj_num = len(obj_list)

        diff_thres = 0
        hl_thres = 120

        cur_mask = np.zeros_like(init_bgr)
        trace_mask = np.zeros_like(init_bgr)
        bgr_with_mask = np.zeros_like(init_bgr)

        for obj in obj_list:
            max_pt_num = max(max_pt_num, len(obj['pt_list']))

        trail_len = 20
        pre_img_idx = -1
        for i in range(0, max_pt_num):
            cur_mask.fill(np.uint8(0))
            for obj in obj_list:
                if i < len(obj['pt_list']):
                    obj_color = np.mean(obj['patch_list'][i], axis=(0,1))
                    obj_color = np.array((np.asscalar(np.uint8(obj_color[0])),np.asscalar(np.uint8(obj_color[1])),np.asscalar(np.uint8(obj_color[2]))))
                    # print obj_color
                    trail_end = i-trail_len if i >= trail_len else 0
                    for j in range(i, trail_end, -1):
                        cur_mask = cv2.line(cur_mask, obj['pt_list'][j],obj['pt_list'][j-1], obj_color, 3)

            bgr = cv2.resize( cv2.imread(self.img_dir[i], flags=1), (0,0), fx=self.resize_rate, fy=self.resize_rate )
            bgr_with_mask = np.where(np.expand_dims(cv2.cvtColor(cur_mask, cv2.COLOR_BGR2GRAY)>cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 
                                                                        axis=3), np.uint8(cur_mask), bgr)
            cv2.imshow('mask', bgr_with_mask)
            if write:
                out.write(bgr_with_mask)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cv2.imshow('mask', bgr_with_mask)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_dir = './denseflow/MAH00399'
    img_dir = sorted(glob.glob(video_dir+'/img_*.jpg'))[65:-20]
    # video_dir = './denseflow/MAH00398'
    # img_dir = sorted(glob.glob(video_dir+'/img_*.jpg'))[200:]
    image = cv2.resize( cv2.imread(img_dir[0], flags=1), (0,0), fx=0.5, fy=0.5)

    # Multiple objects
    mouseReader = MouseClickCollector(image)
    obj_init_centroids = mouseReader.getMouseClick(onlyOneClick=True)

    obj_list = []
    # cur_time = time.time()
    for centroid in obj_init_centroids:
        tracker = TemplateTracker(img_dir, 'point', init_pt=centroid)
        pt_list, reg_list, patch_list = tracker.track(show=False)

        traceGen = TraceGenerator(img_dir)
        new_pt_list = traceGen.interpolateTrace(pt_list, smooth_trace=False , show=False)

        obj_list.append({'pt_list':pt_list, 'new_pt_list':new_pt_list, 'reg_list':reg_list, 'patch_list':patch_list})

    # print time.time() - cur_time
    renderer = TraceRenderer(img_dir)
    renderer.multi_render(obj_list, write=False)
    # renderer.multi_render_drawline(obj_list, write=False)
