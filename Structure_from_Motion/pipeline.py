#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A module containing the pipeline used to achieve structure from motion to yield a point cloud """

import cv2
import numpy as np
import sys
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def pipeline(path_to_dataset, verbose=False):
    """Executes the entire pipeline

    The pipeline can also be executed one step at a time by calling the
    individual functions in order.

    :param verbose: whether output should be produced at each
        processing step (True), or only at the end (False)

    """
    # Create the image loader
    img_loader = Image_loader(path_to_dataset, verbose)
    # load the first image into img2 (i.e. pretend we just processed this image in a previous pair)
    img2 = img_loader.next()
    # Extract feature points from this image
    pts2, desc2 = get_feature_points(img2, verbose)
    # Loop over all the other images
    for count in range(1, img_loader.count):
        # Move the last image (img2) into img1, and its points into pts1
        img1 = img2
        pts1, desc1 = pts2, desc2

        # Load the next image into img2
        img2 = img_loader.next()

        # Extract feature points of the second image
        pts2, desc2 = get_feature_points(img2, verbose)

        # Find the feature point matches between img1 and img2
        matches = match_feature_points(img1, pts1, desc1, img2, pts2, desc2, verbose)
        # Stop after the first image, for testing
        return



class Image_loader:
    """Image_loader

        This class handles loading images to use as input to the pipeline.

        Once initiated, the next image is loaded through next().

    """
    def __init__(self, path_to_dataset, verbose=False):
        self.path = path_to_dataset
        self.index = 0
        # Create an alphabetical list of all the images in the given directory
        with os.scandir(path_to_dataset) as file_iterator:
            self.images = sorted([file_object.name for file_object in list(file_iterator)])
        self.count = len(self.images)

    def next(self):
        """Loads the next image from the given directory

        The images are loaded in alphabetical order, one at a time.
        """
        img = cv2.imread(self.path + self.images[self.index])
        self.index += 1
        return img
    
    def reset(self, new_index=0):
        """Resets the index counter

        Sets the index counter to the given value.

        :param new_index: The index to reset to. Default 0.
        """
        index = new_index

def get_feature_points(img, verbose=False, image_name='Img'):
    """Extract feature points

    Extracts SURF feature points from the given image.

    :param img: the image from which to extract feature points
    :param verbose: as in pipeline()
    :param image_name: the name of the image to be displayed
    """
    # Create the SURF feature detector
    detector = cv2.xfeatures2d.SURF_create(350)
    # Find the keypoints and descriptors in the image
    kp, desc = detector.detectAndCompute(img, None)

    if verbose:
        # Display the image with its feature points
        disp_img = np.zeros_like(img)
        cv2.drawKeypoints(img, kp, disp_img)
        # Shrink the image so it can be displayed in a sensical way
        disp_img = downsize_img(disp_img)
        print(disp_img.shape)
        cv2.imshow(image_name, disp_img)
        cv2.waitKey()
    # Return the keypoints and descriptors
    return kp, desc

def match_feature_points(img1, pts1, desc1, img2, pts2, desc2, verbose=False, image_name='Feature point matches'):
    """Match feature points in the two images

    This should only be called on consecutive images, or correct
    matches are unlikely (or impossible) to be found.

    :param img1: the first image
    :param pts1: feature points from the first image (should be the
        first image read)
    :param desc1: the feature point descriptors of points in pts1
    :param img2: the second image
    :param pts2: feature points from the second image
    :param desc2: the feature point descriptors of points in pts2
    :param verbose: as in pipeline()
    :param image_name: the name of the image to be displayed
    """
    # Create a feature point matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Find the top 2 matches for each feature point
    matches = matcher.knnMatch(desc1, desc2, k=2)
    # Apply the ratio test: x is best match, y is second best match, lower distance is better
    matches = [[x] for x, y in matches if x.distance < 0.75*y.distance]

    if verbose:
        # Display the feature point matches between the two images
        disp_img = np.hstack((img1, img2))
        cv2.drawMatchesKnn(img1, pts1, img2, pts2, matches, disp_img, flags=2)
        disp_img = downsize_img(disp_img)
        cv2.imshow(image_name, disp_img)
        cv2.waitKey()
    # Return the list of matches
    return matches

def get_fundamental_matrix(pt_matches, verbose=False):
    """Estimate the fundamental matrix

    NOTE: this is only valid for this pair of images

    :param pt_matches: the list of feature point matches. These should
        be from consecutive images
    :param verbose: as in pipeline()
    """
    pass

def get_essential_matrix(fmatrix, k, verbose=False):
    """Compute the essential matrix

    :param fmatrix: the fundamental matrix corresponding to this
        essential matrix
    :param k: the camera intrinsics relevant to the two images
        (it is assumed that this will be the same)
    :param verbose: as in pipeline()
    """
    pass

def get_relative_rotation_translation(ematrix, verbose=False):
    """Compute the [R|t] matrix for img2/cam2 from img1/cam1

    :param ematrix: the essential matrix corresponding to these two
        images/cameras
    :param verbose: as in pipeline()
    """
    pass

def get_global_rotation_translation(global_Rt_list, verbose=False):
    """Compute the [R|t] matrix for img2/cam2 from the first img/cam

    :param global_Rt_list: A list of all the global [R|t] matrices for all
        previous images, in order
    :param verbose: as in pipeline()
    """
    pass

def triangulate_feature_points(Rt_list, pt_matches, verbose=False):
    """Get 3D points through triangulation

    :param global_Rt_list: A list of all the global [R|t] matrices for all
        images processed so far, in order
    :param pt_matches: the list of feature point matches between img1 and img2
    :param verbose: as in pipeline()
    """
    pass

#TODO: if the above function only finds relative/local 3D position, this function is needed
# def adjust_feature_points():
#     """

#     """
#     pass

def plot_point_cloud(pts3D, verbose=False):
    """Create a point cloud of all 3D points found so far

    :param pts3D: a list of 3D points to be plotted
    :param verbose: as in pipeline()
    """
    pass

#TODO
def apply_bundle_adjustment():
    """Apply bundle adjustment

    This improves the resultant 3D points through bundle adjustment.

    :param pts3D: the 3D points to be adjusted
    :param verbose: as in pipeline()
    """
    pass

def downsize_img(img, wmax=1600, hmax=900):
    """ Downsize the given image

    Reduce the size of the input image to make it suitable for output.
    The aspect ratio of the image is not changed.

    :param img: the image to downsize
    :param wmax: the maximum acceptable width
    :param hmax: the maximum acceptable height
    """
    # Downsize img until its width is less than wmax and height is less than hmax
    while img.shape[1] > wmax or img.shape[0] > hmax:
        img = cv2.resize(img, None, fx=0.75, fy=0.75)
    return img

# BELOW: left in for reference while constructing the new pipeline above

# Original file copied from https://github.com/mbeyeler/opencv-python-blueprints/tree/master/chapter4
# and edited to fit the task of the assignment

# class Pipeline:
#     """Pipeline

#         This class implements an algorithm for 3D scene reconstruction using
#         stereo vision and structure-from-motion techniques.

#         A 3D scene is reconstructed from a sequence of images captured from
#         the same camera. Feature matching is performed with rich feature
#         descriptors. 3D coordinates are obtained via triangulation.
#     """
#     def __init__(self, K, dist):
#         """Constructor

#             This method initializes the scene reconstruction algorithm.

#             :param K: 3x3 intrinsic camera matrix
#             :param dist: vector of distortion coefficients
#         """
#         self.K = K
#         self.K_inv = np.linalg.inv(K)  # store inverse for fast access
#         self.d = dist

#     def load_image_pair(self, img_path1, img_path2, use_pyr_down=True):
#         """Loads pair of images

#             This method loads the two images for which the 3D scene should be
#             reconstructed. The two images should show the same real-world scene
#             from two different viewpoints.

#             :param img_path1: path to first image
#             :param img_path2: path to second image
#             :param use_pyr_down: flag whether to downscale the images to
#                                  roughly 600px width (True) or not (False)
#         """
#         self.img1 = cv2.imread(img_path1, cv2.CV_8UC3)
#         self.img2 = cv2.imread(img_path2, cv2.CV_8UC3)

#         # make sure images are valid
#         if self.img1 is None:
#             sys.exit("Image " + img_path1 + " could not be loaded.")
#         if self.img2 is None:
#             sys.exit("Image " + img_path2 + " could not be loaded.")

#         if len(self.img1.shape) == 2:
#             self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
#             self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)

#         # scale down image if necessary
#         # to something close to 600px wide
#         target_width = 600
#         if use_pyr_down and self.img1.shape[1] > target_width:
#             while self.img1.shape[1] > 2*target_width:
#                 self.img1 = cv2.pyrDown(self.img1)
#                 self.img2 = cv2.pyrDown(self.img2)

#         # undistort the images
#         self.img1 = cv2.undistort(self.img1, self.K, self.d)
#         self.img2 = cv2.undistort(self.img2, self.K, self.d)

#     def plot_optic_flow(self):
#         """Plots optic flow field

#             This method plots the optic flow between the first and second
#             image.
#         """
#         self._extract_keypoints("flow")

#         img = self.img1
#         for i in range(len(self.match_pts1)):
#             cv2.line(img, tuple(self.match_pts1[i]), tuple(self.match_pts2[i]),
#                      color=(255, 0, 0))
#             theta = np.arctan2(self.match_pts2[i][1] - self.match_pts1[i][1],
#                                self.match_pts2[i][0] - self.match_pts1[i][0])
#             cv2.line(img, tuple(self.match_pts2[i]),
#                      (np.int(self.match_pts2[i][0] - 6*np.cos(theta+np.pi/4)),
#                       np.int(self.match_pts2[i][1] - 6*np.sin(theta+np.pi/4))),
#                      color=(255, 0, 0))
#             cv2.line(img, tuple(self.match_pts2[i]),
#                      (np.int(self.match_pts2[i][0] - 6*np.cos(theta-np.pi/4)),
#                       np.int(self.match_pts2[i][1] - 6*np.sin(theta-np.pi/4))),
#                      color=(255, 0, 0))

#         cv2.imshow("imgFlow", img)
#         cv2.waitKey()

#     def draw_epipolar_lines(self, feat_mode="SURF"):
#         """Draws epipolar lines

#             This method computes and draws the epipolar lines of the two
#             loaded images.

#             :param feat_mode: whether to use rich descriptors for feature
#                               matching ("surf") or optic flow ("flow")
#         """
#         self._extract_keypoints(feat_mode)
#         self._find_fundamental_matrix()
#         # Find epilines corresponding to points in right image (second image)
#         # and drawing its lines on left image
#         pts2re = self.match_pts2.reshape(-1, 1, 2)
#         lines1 = cv2.computeCorrespondEpilines(pts2re, 2, self.F)
#         lines1 = lines1.reshape(-1, 3)
#         img3, img4 = self._draw_epipolar_lines_helper(self.img1, self.img2,
#                                                       lines1, self.match_pts1,
#                                                       self.match_pts2)

#         # Find epilines corresponding to points in left image (first image) and
#         # drawing its lines on right image
#         pts1re = self.match_pts1.reshape(-1, 1, 2)
#         lines2 = cv2.computeCorrespondEpilines(pts1re, 1, self.F)
#         lines2 = lines2.reshape(-1, 3)
#         img1, img2 = self._draw_epipolar_lines_helper(self.img2, self.img1,
#                                                       lines2, self.match_pts2,
#                                                       self.match_pts1)

#         cv2.imshow("left", img1)
#         cv2.imshow("right", img3)
#         cv2.waitKey()

#     def plot_rectified_images(self, feat_mode="SURF"):
#         """Plots rectified images

#             This method computes and plots a rectified version of the two
#             images side by side.

#             :param feat_mode: whether to use rich descriptors for feature
#                               matching ("surf") or optic flow ("flow")
#         """
#         self._extract_keypoints(feat_mode)
#         self._find_fundamental_matrix()
#         self._find_essential_matrix()
#         self._find_camera_matrices_rt()

#         R = self.Rt2[:, :3]
#         T = self.Rt2[:, 3]
#         #perform the rectification
#         R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K, self.d,
#                                                           self.K, self.d,
#                                                           self.img1.shape[:2],
#                                                           R, T, alpha=1.0)
#         mapx1, mapy1 = cv2.initUndistortRectifyMap(self.K, self.d, R1, self.K,
#                                                    self.img1.shape[:2],
#                                                    cv2.CV_32F)
#         mapx2, mapy2 = cv2.initUndistortRectifyMap(self.K, self.d, R2, self.K,
#                                                    self.img2.shape[:2],
#                                                    cv2.CV_32F)
#         img_rect1 = cv2.remap(self.img1, mapx1, mapy1, cv2.INTER_LINEAR)
#         img_rect2 = cv2.remap(self.img2, mapx2, mapy2, cv2.INTER_LINEAR)

#         # draw the images side by side
#         total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
#                       img_rect1.shape[1] + img_rect2.shape[1], 3)
#         img = np.zeros(total_size, dtype=np.uint8)
#         img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
#         img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

#         # draw horizontal lines every 25 px accross the side by side image
#         for i in range(20, img.shape[0], 25):
#             cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

#         cv2.imshow('imgRectified', img)
#         cv2.waitKey()

#     def plot_point_cloud(self, feat_mode="SURF"):
#         """Plots 3D point cloud

#             This method generates and plots a 3D point cloud of the recovered
#             3D scene.

#             :param feat_mode: whether to use rich descriptors for feature
#                               matching ("surf") or optic flow ("flow")
#         """
#         self._extract_keypoints(feat_mode)
#         self._find_fundamental_matrix()
#         self._find_essential_matrix()
#         self._find_camera_matrices_rt()

#         # triangulate points
#         first_inliers = np.array(self.match_inliers1).reshape(-1, 3)[:, :2]
#         second_inliers = np.array(self.match_inliers2).reshape(-1, 3)[:, :2]
#         pts4D = cv2.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T,
#                                       second_inliers.T).T

#         # convert from homogeneous coordinates to 3D
#         pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

#         # plot with matplotlib
#         Ys = pts3D[:, 0]
#         Zs = pts3D[:, 1]
#         Xs = pts3D[:, 2]

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(Xs, Ys, Zs, c='r', marker='o')
#         ax.set_xlabel('Y')
#         ax.set_ylabel('Z')
#         ax.set_zlabel('X')
#         plt.title('3D point cloud: Use pan axes button below to inspect')
#         plt.show()

#     def _extract_keypoints(self, feat_mode):
#         """Extracts keypoints

#             This method extracts keypoints for feature matching based on
#             a specified mode:
#             - "surf": use rich SURF descriptor
#             - "flow": use optic flow

#             :param feat_mode: keypoint extraction mode ("surf" or "flow")
#         """
#         # extract features
#         if feat_mode.lower() == "surf":
#             # feature matching via SURF and BFMatcher
#             self._extract_keypoints_surf()
#         else:
#             if feat_mode.lower() == "flow":
#                 # feature matching via optic flow
#                 self._extract_keypoints_flow()
#             else:
#                 sys.exit("Unknown feat_mode " + feat_mode +
#                          ". Use 'SURF' or 'FLOW'")

#     def _extract_keypoints_surf(self):
#         """Extracts keypoints via SURF descriptors"""
#         # extract keypoints and descriptors from both images
#         detector = cv2.xfeatures2d.SURF_create(250)

#         first_key_points, first_desc = detector.detectAndCompute(self.img1,
#                                                                  None)
#         second_key_points, second_desc = detector.detectAndCompute(self.img2,
#                                                                    None)

#         # match descriptors
#         matcher = cv2.BFMatcher(cv2.NORM_L1, True)
#         matches = matcher.match(first_desc, second_desc)

#         # generate lists of point correspondences
#         first_match_points = np.zeros((len(matches), 2), dtype=np.float32)
#         second_match_points = np.zeros_like(first_match_points)
#         for i in range(len(matches)):
#             first_match_points[i] = first_key_points[matches[i].queryIdx].pt
#             second_match_points[i] = second_key_points[matches[i].trainIdx].pt

#         self.match_pts1 = first_match_points
#         self.match_pts2 = second_match_points

#     def _extract_keypoints_flow(self):
#         """Extracts keypoints via optic flow"""
#         # find FAST features
#         fast = cv2.FastFeatureDetector()
#         first_key_points = fast.detect(self.img1, None)

#         first_key_list = [i.pt for i in first_key_points]
#         first_key_arr = np.array(first_key_list).astype(np.float32)

#         second_key_arr, status, err = cv2.calcOpticalFlowPyrLK(self.img1,
#                                                                self.img2,
#                                                                first_key_arr)

#         # filter out the points with high error
#         # keep only entries with status=1 and small error
#         condition = (status == 1) * (err < 5.)
#         concat = np.concatenate((condition, condition), axis=1)
#         first_match_points = first_key_arr[concat].reshape(-1, 2)
#         second_match_points = second_key_arr[concat].reshape(-1, 2)

#         self.match_pts1 = first_match_points
#         self.match_pts2 = second_match_points

#     def _find_fundamental_matrix(self):
#         """Estimates fundamental matrix """
#         # Uses RANSAC to produce an improved estimate of the fundamental matrix
#         self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
#                                                     self.match_pts2,
#                                                     cv2.FM_RANSAC, 0.1, 0.99)

#     def _find_essential_matrix(self):
#         """Estimates essential matrix based on fundamental matrix """
#         self.E = self.K.T.dot(self.F).dot(self.K)

#     def _find_camera_matrices_rt(self):
#         """Finds the [R|t] camera matrix"""
#         # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
#         U, S, Vt = np.linalg.svd(self.E)
#         W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
#                       1.0]).reshape(3, 3)

#         # iterate over all point correspondences used in the estimation of the
#         # fundamental matrix
#         first_inliers = []
#         second_inliers = []
#         for i in range(len(self.Fmask)):
#             if self.Fmask[i]:
#                 # normalize and homogenize the image coordinates
#                 first_inliers.append(self.K_inv.dot([self.match_pts1[i][0],
#                                      self.match_pts1[i][1], 1.0]))
#                 second_inliers.append(self.K_inv.dot([self.match_pts2[i][0],
#                                       self.match_pts2[i][1], 1.0]))

#         # Determine the correct choice of second camera matrix
#         # only in one of the four configurations will all the points be in
#         # front of both cameras
#         # First choice: R = U * W * Vt, T = +u_3 (See Hartley Zisserman 9.19)
#         R = U.dot(W).dot(Vt)
#         T = U[:, 2]
#         if not self._in_front_of_both_cameras(first_inliers, second_inliers,
#                                               R, T):
#             # Second choice: R = U * W * Vt, T = -u_3
#             T = - U[:, 2]

#         if not self._in_front_of_both_cameras(first_inliers, second_inliers,
#                                               R, T):
#             # Third choice: R = U * Wt * Vt, T = u_3
#             R = U.dot(W.T).dot(Vt)
#             T = U[:, 2]

#             if not self._in_front_of_both_cameras(first_inliers,
#                                                   second_inliers, R, T):
#                 # Fourth choice: R = U * Wt * Vt, T = -u_3
#                 T = - U[:, 2]

#         self.match_inliers1 = first_inliers
#         self.match_inliers2 = second_inliers
#         self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
#         self.Rt2 = np.hstack((R, T.reshape(3, 1)))
#         # Print Rt
#         print("[R|t] matrix:")
#         print(self.Rt2)

#     def _draw_epipolar_lines_helper(self, img1, img2, lines, pts1, pts2):
#         """Helper method to draw epipolar lines and features """
#         if img1.shape[2] == 1:
#             img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#         if img2.shape[2] == 1:
#             img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

#         c = img1.shape[1]
#         for r, pt1, pt2 in zip(lines, pts1, pts2):
#             color = tuple(np.random.randint(0, 255, 3).tolist())
#             x0, y0 = map(int, [0, -r[2]/r[1]])
#             x1, y1 = map(int, [c, -(r[2] + r[0]*c) / r[1]])
#             cv2.line(img1, (x0, y0), (x1, y1), color, 1)
#             cv2.circle(img1, tuple(pt1), 5, color, -1)
#             cv2.circle(img2, tuple(pt2), 5, color, -1)
#         return img1, img2

#     def _in_front_of_both_cameras(self, first_points, second_points, rot,
#                                   trans):
#         """Determines whether point correspondences are in front of both
#            images"""
#         rot_inv = rot
#         for first, second in zip(first_points, second_points):
#             first_z = np.dot(rot[0, :] - second[0]*rot[2, :],
#                              trans) / np.dot(rot[0, :] - second[0]*rot[2, :],
#                                              second)
#             first_3d_point = np.array([first[0] * first_z,
#                                        second[0] * first_z, first_z])
#             second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
#                                                                      trans)

#             if first_3d_point[2] < 0 or second_3d_point[2] < 0:
#                 return False

#         return True

#    def _linear_ls_triangulation(self, u1, P1, u2, P2):
#        """Triangulation via Linear-LS method"""
#        # build A matrix for homogeneous equation system Ax=0
#        # assume X = (x,y,z,1) for Linear-LS method
#        # which turns it into AX=B system, where A is 4x3, X is 3x1 & B is 4x1
#        A = np.array([u1[0]*P1[2, 0] - P1[0, 0], u1[0]*P1[2, 1] - P1[0, 1],
#                      u1[0]*P1[2, 2] - P1[0, 2], u1[1]*P1[2, 0] - P1[1, 0],
#                      u1[1]*P1[2, 1] - P1[1, 1], u1[1]*P1[2, 2] - P1[1, 2],
#                      u2[0]*P2[2, 0] - P2[0, 0], u2[0]*P2[2, 1] - P2[0, 1],
#                      u2[0]*P2[2, 2] - P2[0, 2], u2[1]*P2[2, 0] - P2[1, 0],
#                      u2[1]*P2[2, 1] - P2[1, 1],
#                      u2[1]*P2[2, 2] - P2[1, 2]]).reshape(4, 3)

#        B = np.array([-(u1[0]*P1[2, 3] - P1[0, 3]),
#                      -(u1[1]*P1[2, 3] - P1[1, 3]),
#                      -(u2[0]*P2[2, 3] - P2[0, 3]),
#                      -(u2[1]*P2[2, 3] - P2[1, 3])]).reshape(4, 1)

#        ret, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
#        return X.reshape(1, 3)