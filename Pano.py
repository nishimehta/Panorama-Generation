#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:16:50 2018

@author: nishimehta
"""
#%%
import numpy as np
import cv2 as cv2

UBIT = '50291671' 
np.random.seed(sum([ord(c) for c in UBIT]))


#%%
print('Task 1')
print('Task 1.1')
MIN_MATCH_COUNT = 10
img1 = cv2.imread('mountain1.jpg')          
img2 = cv2.imread('mountain2.jpg') 
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

sift_img1=cv2.drawKeypoints(img1,kp1,None,color = (255,0,0))
cv2.imwrite('task1_sift1.jpg',sift_img1)
sift_img2=cv2.drawKeypoints(img2,kp2,None,color = (255,0,0))
cv2.imwrite('task1_sift2.jpg',sift_img2)

#%%
print('Task 1.2')
BF = cv2.BFMatcher()
matches = BF.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
        
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
           singlePointColor = None,
           flags = 2)
matches_knn = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite('task1_matches_knn.jpg',matches_knn)
        
#%%
print('Task 1.3')
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
print('Homography Matrix:')
print(H)
#%%    
print('Task 1.4')
matchesMask = mask.ravel().tolist()

inliers=[]
for i in range(len(matchesMask)):
    if(matchesMask[i]==1):
        inliers.append(good[i])
#randomly pick 10 inliers
random_10_matches = np.random.choice(inliers,10)
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
randomMatchesImg = cv2.drawMatches(img1,kp1,img2,kp2,random_10_matches,None,**draw_params)
cv2.imwrite('task1_matches.jpg',randomMatchesImg)

#%%
print('Task 1.5')
#perspective transformation inorder to keep all pixels
rows1, cols1 = img1.shape[:2]
rows2, cols2 = img2.shape[:2]

corners_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
corners_2 = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

perspective_corners_1 = cv2.perspectiveTransform(corners_1, H)
corners = np.concatenate((corners_2, perspective_corners_1), axis=0)

[x_min, y_min] = np.int32(corners.min(axis=0).ravel())
[x_max, y_max] = np.int32(corners.max(axis=0).ravel())

TY = -y_min
TX = -x_min
T = np.asarray([[1,0,TX],[0,1,TY],[0,0,1]])
H_T = np.dot(T,H)
im_out = cv2.warpPerspective(img1,H_T, (x_max-x_min,y_max-y_min))
#copying image 2 after warping image 1
im_out[TY:TY+rows2,TX:TX+cols2] = img2
        
cv2.imwrite('task1_pano.jpg',im_out)
