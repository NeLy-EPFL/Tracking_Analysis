#!/usr/bin/env python

import math
import cv2
import skimage
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from pathlib import Path

import platform

platform.system()

#Working on mac
#im_bg_dir_path = Path("/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/TrackingFiles/BackgroundImages/")
#im_fg_dir_path = Path("/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/TrackingFiles/ForegroundImages/")
#vid_dir_path = Path("/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/TrackingFiles/Sample/")
#save_data_path = Path("/Volumes/Ramdya-Lab/DURRIEU_Matthias/Experimental_data/TrackingFiles/Processed/")

#Working on workstation
im_bg_dir_path = Path("/mnt/labserver/DURRIEU_Matthias/Experimental_data/TrackingFiles/BackgroundImages/")
im_fg_dir_path = Path("/mnt/labserver/DURRIEU_Matthias/Experimental_data/TrackingFiles/ForegroundImages/")
vid_dir_path = Path("/mnt/labserver/DURRIEU_Matthias/Experimental_data/TrackingFiles/Sample/")
save_data_path = Path("/mnt/labserver/DURRIEU_Matthias/Experimental_data/TrackingFiles/Processed/")

#if (platform.system() == 'Windows'):
   # im_bg_dir_path = "C:\\Users\\123mn\\OneDrive\\Documents\\0_Assistant_temp_EPFL\\Optobot\\optobot_control_ROS\\data\\background_elem\\"
    #im_fg_dir_path = "C:\\Users\\123mn\\OneDrive\\Documents\\0_Assistant_temp_EPFL\\Optobot\\optobot_control_ROS\\data\\foreground_elem\\"

   # vid_dir_path = "C:\\Users\\123mn\\OneDrive\\Documents\\0_Assistant_temp_EPFL\\Optobot\\optobot_control_ROS\\data\\videos\\"

    #save_data_path = "C:\\Users\\123mn\\OneDrive\\Documents\\0_Assistant_temp_EPFL\\Optobot\\optobot_control_ROS\\data\\saved\\"
#else:
    #im_bg_dir_path = "/home/nely/optobot_control_ROS/data/background_elem/"
   # im_fg_dir_path = "/home/nely/optobot_control_ROS/data/foreground_elem/"

    #vid_dir_path = "/home/nely/optobot_control_ROS/data/videos/"

    #save_data_path = "/home/nely/optobot_control_ROS/data/saved/"

# parameters
rescaling_factor = 4
nb_balls = 3
nb_rails = 3


# In[1]: Clean images

# hard to retain only the outer edges
def im_clean_opening(im, op_kernel=np.ones((3, 3), np.uint8), bin_thres=80, blur_kernel=(3, 3)):
    # im_blur = cv2.GaussianBlur(im,blur_kernel,0)
    ret, im_bin = cv2.threshold(im, bin_thres, 255, cv2.THRESH_BINARY)

    im_bin_ero = cv2.erode(im_bin, op_kernel, iterations=1)

    im_bin_op = cv2.dilate(im_bin_ero, op_kernel, iterations=1)

    return im_bin_op


# takes a lot of time
def im_clean_region_growing(im, tol=40, seed_thres=80, dil_kernel=np.ones((13, 13), np.uint8)):
    region = []
    region_nb = 0

    for i, j in np.ndindex(im.shape):
        region_labelled_b = False
        if im[i][j] > seed_thres:
            for k in range(region_nb):
                if region[k][i][j] == 1:
                    region_labelled_b = True
            if region_labelled_b == False:
                region.append(skimage.segmentation.flood(im, (i, j), tolerance=tol))
                region_nb += 1
        if region_nb > 1000:
            print("region growing break")
            break

    nb_pix = []
    for k in range(region_nb):
        nb_pix.append(len(np.where(region[k])[0]))

    max_idx = np.argmax(nb_pix)

    outer_arena_mask = np.array(region[max_idx], dtype=np.uint8)
    outer_arena_mask_dil = cv2.dilate(outer_arena_mask, dil_kernel, iterations=1)
    im_clean = im.copy()
    im_clean[outer_arena_mask_dil == 0] = 0

    return im_clean


def im_clean_contours(im, mode, dil_kernel=np.ones((2, 2), np.uint8), bin_thresh=50):
    # apply binary thresholding
    ret, thresh = cv2.threshold(im, bin_thresh, 255, cv2.THRESH_BINARY)

    # plt.figure()
    # plt.imshow(thresh)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[-2:]

    if mode == "single_contour":
        contours_len = []
        # finding longest contour
        for i in range(len(contours)):
            contours_len.append(len(contours[i]))
        contours_max_len_idx = np.argmax(contours_len)

        mask = np.zeros(im.shape)
        # draw contours on the original image
        mask = cv2.drawContours(image=mask, contours=contours, contourIdx=contours_max_len_idx,
                                color=1, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
    elif mode == "in_bet_two_contours":
        contours_len = []

        # finding longest contour
        for i in range(len(contours)):
            contours_len.append(len(contours[i]))
        contours_max_len_idx = np.argmax(contours_len)
        contours_len[contours_max_len_idx] = 0

        contours_sec_max_len_idx = np.argmax(contours_len)

        outer_mask = np.zeros(im.shape)
        outer_mask = cv2.drawContours(image=outer_mask, contours=contours, contourIdx=contours_max_len_idx,
                                      color=1, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        inner_mask = np.zeros(im.shape)
        inner_mask = cv2.drawContours(image=inner_mask, contours=contours, contourIdx=contours_sec_max_len_idx,
                                      color=1, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

        # plt.figure()
        # plt.imshow(outer_mask)
        # plt.figure()
        # plt.imshow(inner_mask)

        mask = cv2.subtract(outer_mask, inner_mask)

    mask_dil = cv2.dilate(mask, dil_kernel, iterations=1)

    im_clean = im.copy()
    im_clean[mask_dil == 0] = 0

    return im_clean


# In[2]: Template matching

'''patented
def compute_homography_surf(im_obj, im_scene, minHessian=100, ratio_thresh=0.9): 
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors    
    #minHessian lower -> more points #100 for arena
    #ratio thresh 0.9 for arena
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold=minHessian, extended=False)

    keypoints_obj, descriptors_obj = detector.detectAndCompute(im_obj, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(im_scene, None)
    #print(np.shape(keypoints_obj))

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    #print(np.shape(knn_matches))
    #print(knn_matches)

    #-- Filter matches using the Lowe's ratio test
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    #H, _ =  cv2.findHomography(obj, scene, cv2.RHO) #RANSAC
    #H, _ =  cv2.estimateAffine2D(obj, scene, cv2.RANSAC)
    H, _ =  cv2.estimateAffinePartial2D(obj, scene, cv2.RANSAC)

    return H
'''

''' fast only detector, not descriptor => would need BRISK or oder
def compute_homography_fast(im_obj, im_scene):
    detector = cv2.FastFeatureDetector_create()

    keypoints_obj = detector.detect(im_obj, None)
    keypoints_scene = detector.detect(im_scene, None)
    #print(np.shape(keypoints_obj))

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    #print(np.shape(knn_matches))
    #print(knn_matches)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.85
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    #H, _ =  cv2.findHomography(obj, scene, cv2.RHO) #RANSAC
    #H, _ =  cv2.estimateAffine2D(obj, scene, cv2.RANSAC)
    H, _ =  cv2.estimateAffinePartial2D(obj, scene, cv2.RANSAC)

    return H
'''


def compute_homography_orb(im_obj, im_scene):
    detector = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=4,
                              edgeThreshold=15, firstLevel=0, WTA_K=2,
                              patchSize=15,
                              fastThreshold=10)  # nlevels=1 : supposedly (according to forums) no scaling but not the case

    keypoints_obj, descriptors_obj = detector.detectAndCompute(im_obj, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(im_scene, None)
    # print(np.shape(keypoints_obj))

    # find matches
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(descriptors_obj, descriptors_scene)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    number_of_matches = 400

    # calculate transformation matrix
    base_keypoints = np.float32([keypoints_obj[m.queryIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)
    test_keypoints = np.float32([keypoints_scene[m.trainIdx].pt for m in matches[:number_of_matches]]).reshape(-1, 1, 2)
    # Calculate Homography
    # H, _ =  cv2.findHomography(obj, scene, cv2.RHO) #RANSAC
    # H, _ =  cv2.estimateAffine2D(obj, scene, cv2.RANSAC)
    H, _ = cv2.estimateAffinePartial2D(base_keypoints, test_keypoints, cv2.RANSAC)

    return H


'''patented
def compute_homography_sift(im_obj, im_scene):
    #-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors    
    detector = cv2.xfeatures2d.SIFT_create()

    keypoints_obj, descriptors_obj = detector.detectAndCompute(im_obj, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(im_scene, None)
    #print(np.shape(keypoints_obj))

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.85
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    #H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)
    #H, _ =  cv2.estimateAffine2D(obj, scene, cv2.RANSAC)
    H, _ =  cv2.estimateAffinePartial2D(obj, scene, cv2.RANSAC)
    return H

#not working (at least for rails)
def compute_homography_euclidean(im_obj, im_scene):
    #-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors    
    detector = cv2.xfeatures2d.SIFT_create()

    keypoints_obj, descriptors_obj = detector.detectAndCompute(im_obj, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(im_scene, None)
    #print(np.shape(keypoints_obj))

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.85 #0.85 for arena
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    #H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)
    #H, _ =  cv2.estimateAffine2D(obj, scene, cv2.RANSAC)
    #H, _ =  cv2.estimateAffinePartial2D(obj, scene, cv2.RANSAC)
    tform3 = skimage.transform.EuclideanTransform()
    H = tform3.estimate(obj, scene)
    #print(H)

    #plt.figure()
    #plt.imshow(im_obj)

    warped = skimage.transform.warp(im_obj, tform3)
    #plt.figure()
    #plt.imshow(warped)

    return H
'''


def compute_gen_im_template(im_gray, H, im_nb):
    im_size = np.shape(im_gray[0])

    # transformation with respect to the first one
    im_bg_gray_gen_avg = im_gray[0]

    for i in range(1, im_nb):
        idx = i - 1

        # im_bg_gray_gen_temp = cv2.warpPerspective(im_gray[i], H[idx], im_size)
        im_bg_gray_gen_temp = cv2.warpAffine(im_gray[i], H[idx], im_size)
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(im_gray[i], cmap='gray', vmin=0, vmax=255)
        # plt.subplot(1,3,2)
        # plt.imshow(im_bg_gray_gen_temp, cmap='gray', vmin=0, vmax=255)

        # performs averaging
        alpha = 1.0 / (i + 1)
        beta = 1.0 - alpha
        im_bg_gray_gen_avg = cv2.addWeighted(im_bg_gray_gen_temp, alpha, im_bg_gray_gen_avg, beta, 0.0)
        # plt.subplot(1,3,3)
        # plt.imshow(im_bg_gray_gen_avg, cmap='gray', vmin=0, vmax=255)

    return im_bg_gray_gen_avg


# In[3]: Axes of inertia

def compute_centroid_angle(im):
    M = cv2.moments(im)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    alpha = (1 / 2 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))) * 180 / np.pi
    return cx, cy, alpha


def trans_rot_inertia(im_gray, im_size):
    rows, cols = im_size

    cx, cy, alpha = compute_centroid_angle(im_gray)

    dist_to_center_x = int(rows / 2 - cx)
    dist_to_center_y = int(cols / 2 - cy)

    # translation
    M = np.float32([[1, 0, dist_to_center_x], [0, 1, dist_to_center_y]])
    trans = cv2.warpAffine(im_gray, M, (cols, rows))

    # rotation
    M = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), alpha, 1)
    rot = cv2.warpAffine(trans, M, (cols, rows))

    return rot


def compute_gen_im_inertia(im_gray, im_nb):
    im_size = im_gray[0].shape
    for i in range(im_nb):
        if i == 0:
            im_gray_gen_avg = trans_rot_inertia(im_gray[i], im_size)
        else:
            im_bg_gray_gen_temp = trans_rot_inertia(im_gray[i], im_size)
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            im_bg_gray_gen_avg = cv2.addWeighted(im_bg_gray_gen_temp, alpha, im_gray_gen_avg, beta, 0.0)

    return im_bg_gray_gen_avg


# In[4]: Contour matching

def contour_matching(im, contour_to_match, contour_length_thresh=60):
    # _, im_bin = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, im_bin = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
    # plt.figure()
    # plt.imshow(im_bin, cmap='gray', vmin=0, vmax=255)
    im_contours, hierarchy = cv2.findContours(image=im_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[-2:]

    mask = cv2.cvtColor(np.zeros(im.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    mask = cv2.drawContours(image=mask, contours=im_contours, contourIdx=-1,
                            color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    # plt.figure()
    # plt.imshow(mask)

    # discriminated by length (many other criteria possible)
    big_contours = []
    for contour in im_contours:
        if len(contour) > contour_length_thresh:
            big_contours.append(contour)

    mask = cv2.cvtColor(np.zeros(im.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    mask = cv2.drawContours(image=mask, contours=big_contours, contourIdx=-1,
                            color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    # plt.figure()
    # plt.imshow(mask)

    match_val = []
    for contour in big_contours:
        match_val.append(cv2.matchShapes(contour, contour_to_match, 1, 0.0))
    idx = np.argsort(match_val)

    return big_contours[idx[0]][:]


# In[5]: Post-processing

def enhance_center(im, kernel=1 / 12 * np.array([[1, 2, 1],
                                                 [2, 4, 2],
                                                 [1, 2, 1]])):
    return cv2.filter2D(src=im, ddepth=-1, kernel=kernel)


def uniform_illumination(im, factor=2, bin_thresh=35):
    mean_intens = np.mean(im[im > bin_thresh])
    im_unif = im.copy()
    im_unif[im > bin_thresh] = mean_intens * factor
    im_unif = np.array(im_unif, np.uint8)
    return im_unif


# In[6]: Background generation

def gen_background(im_full_gray):
    background_full = np.zeros(im_full_gray.shape, np.uint8)
    background_arena_door = np.zeros(im_full_gray.shape, np.uint8)
    background_rails = np.zeros(im_full_gray.shape, np.uint8)

    # reading images of background elements
    im_arena_door_gray = cv2.cvtColor(cv2.imread(im_bg_dir_path.joinpath("im_arena_door.jpg").as_posix()), cv2.COLOR_BGR2GRAY)
    im_arena_door_gray_unif = cv2.cvtColor(cv2.imread(im_bg_dir_path.joinpath("im_arena_door_unif.jpg").as_posix()), cv2.COLOR_BGR2GRAY)
    im_rail_gray = cv2.cvtColor(cv2.imread(im_bg_dir_path.joinpath("im_rail.jpg").as_posix()), cv2.COLOR_BGR2GRAY)
    im_rail_gray_unif = cv2.cvtColor(cv2.imread(im_bg_dir_path.joinpath("im_rail_unif.jpg").as_posix()), cv2.COLOR_BGR2GRAY)

    # resizing elements
    im_arena_door_gray_resized = cv2.resize(im_arena_door_gray, (im_full_gray.shape[1], im_full_gray.shape[0]),
                                            interpolation=cv2.INTER_AREA)
    im_arena_door_gray_unif_resized = cv2.resize(im_arena_door_gray_unif,
                                                 (im_full_gray.shape[1], im_full_gray.shape[0]),
                                                 interpolation=cv2.INTER_AREA)
    im_rail_gray_resized = cv2.resize(im_rail_gray, (im_full_gray.shape[1], im_full_gray.shape[0]),
                                      interpolation=cv2.INTER_AREA)
    im_rail_gray_unif_resized = cv2.resize(im_rail_gray_unif, (im_full_gray.shape[1], im_full_gray.shape[0]),
                                           interpolation=cv2.INTER_AREA)
    # arena and door
    im_size = im_full_gray.shape
    H_arena_door = compute_homography_orb(im_arena_door_gray_resized, im_full_gray)  # 500, 0.85
    im_arena_door_gray_unif_resized_trans = cv2.warpAffine(im_arena_door_gray_unif_resized, H_arena_door, im_size)

    im_arena_door_gray_unif_resized_trans = cv2.dilate(im_arena_door_gray_unif_resized_trans, np.ones((2, 2), np.uint8))
    background_full = cv2.add(background_full, im_arena_door_gray_unif_resized_trans)
    background_arena_door = cv2.add(background_arena_door, im_arena_door_gray_unif_resized_trans)

    # plt.figure()
    # plt.imshow(background_full)

    im_full_gray_ad = cv2.subtract(im_full_gray, im_arena_door_gray_unif_resized_trans)

    # subtracting rails
    im_full_gray_adr_temp = im_full_gray_ad.copy()
    im_full_gray_adr = im_full_gray_ad.copy()
    im_rail_gray_to_sub = im_rail_gray_unif_resized.copy()

    # finding outer contour of rail template
    _, im_rail_bin = cv2.threshold(im_rail_gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_rail_contours, hierarchy = cv2.findContours(image=im_rail_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[
                                  -2:]

    im_rail_contours_len = []
    # finding longest (outer) contour
    for i in range(len(im_rail_contours)):
        im_rail_contours_len.append(len(im_rail_contours[i]))
    im_rail_contours_max_len_idx = np.argmax(im_rail_contours_len)

    contour_to_match = im_rail_contours[im_rail_contours_max_len_idx]
    rows, cols = im_full_gray_adr_temp.shape

    for i in range(nb_rails):
        contour_found = contour_matching(im_full_gray_adr_temp, contour_to_match)

        mask = np.zeros(im_full_gray_adr_temp.shape)
        mask = cv2.drawContours(image=mask, contours=[contour_found], contourIdx=-1,
                                color=1, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        # plt.figure()
        # plt.imshow(mask)
        # plt.title("mask")

        im_full_gray_adr_temp[mask == 1] = 0

        cx, cy, alpha = compute_centroid_angle(mask)
        # print(cx,cy,alpha)

        # rotation
        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), alpha, 1)
        rot = cv2.warpAffine(im_rail_gray_to_sub, M, (rows, cols))

        dist_to_center_x = cx - rows / 2
        dist_to_center_y = cy - cols / 2
        # translation
        M = np.float32([[1, 0, dist_to_center_x], [0, 1, dist_to_center_y]])
        trans = cv2.warpAffine(rot, M, (rows, cols))

        background_full = cv2.add(background_full, trans)
        background_rails = cv2.add(background_rails, trans)

        im_full_gray_adr = cv2.subtract(im_full_gray_adr, trans)

    return background_arena_door, background_rails, background_full
