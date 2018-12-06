#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A module containing the pipeline used to achieve structure from motion to yield a point cloud """

import cv2
import numpy as np
import sys
import os
import random

import open3d

def pipeline(path_to_dataset, k, verbose=False, verbose_img=False):
    """Executes the entire pipeline

    The pipeline can also be executed one step at a time by calling the
    individual functions in order.

    :param path_to_dataset: the directory containing the dataset. This
        should only contain the images of the dataset, ordered
        alphabetically (i.e. the first image is the first alphabetically)
    :param k: the camera intrinsics matrix of the camera used to capture
        the dataset
    :param verbose: whether output should be produced at each processing
        step (True), or only at the end (False)
    :param verbose_img: whether image output should be produced where
        relevant
    """
    # Create the image loader
    img_loader = Image_loader(path_to_dataset, verbose)

    # Skip to a specific image
    img_loader.reset(6)

    # Set the start index, for printing purposes only
    start_index = img_loader.index
    # load the first image into imgR (i.e. pretend we just processed this image in a previous pair)
    print('Processing image {:3} out of {}'.format(start_index+1, img_loader.count))
    imgR = img_loader.next()

    # Extract feature points from this image
    ptsR, descR = get_feature_points(imgR, verbose=verbose, verbose_img=verbose_img)

    # Initialize some variables used for the entire sequence
    rtmatrix1 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float64).reshape(3, 4) # The global [R|t] matrix for image 1
    global_rt_list = rtmatrix1.reshape(1, 3, 4) # List of the global [R|t] matrices for each image
    all_pts4D = None # List of all 4D points, to be plotted at the end
    pts4D_indices = [0]
    acceptable_z = True # Set this to true initially so that the pipeline initializes properly
                        # This variable is used to exclude point clouds from frames for which no acceptable [R|t] matrix was found
    # Loop over all the other images
    for count in range(1, img_loader.count):
        if count+start_index+1 <= img_loader.count:
            print('Processing image {:3} out of {}'.format(count+start_index+1, img_loader.count))
        # Move the last image (imgR) into imgL, and its points into ptsL
        # Only do this if the last image wasn't skipped - if it was, disregard it and match to the 2nd last image instead
        if acceptable_z:
            imgL = imgR
            ptsL, descL = ptsR, descR

        # Load the next image into imgR
        imgR = img_loader.next()
        # Check that the image was loaded correctly. If not, it will be False
        if type(imgR) == bool and imgR == False:
            # This means there are no more images to load, so break out of the loop and plot the points
            break

        # Extract feature points of the second image
        ptsR, descR = get_feature_points(imgR, verbose=verbose, verbose_img=verbose_img)

        # Find the feature point matches between imgL and imgR
        matches, imgL_matches, imgR_matches = match_feature_points(imgL, ptsL, descL, imgR, ptsR, descR, verbose=verbose, verbose_img=verbose_img)

        # Repeat the below section of the pipeline until an acceptable_z [R|t] matrix is found. If none is found after TRIALS
        # tries, skip this image
        acceptable_z = False
        TRIALS = 30
        current_trial = 0
        while not acceptable_z:
            if current_trial >= TRIALS:
                # Skip this image
                break
            # Estimate the fundamental matrix
            fmatrix, fmap = get_fundamental_matrix(imgL_matches, imgR_matches, verbose=verbose)

            # If no fmatrix was found, skip the following
            if fmatrix != None:

                # Calculate the essential matrix
                ematrix = get_essential_matrix(fmatrix, k, verbose=verbose)

                # Get the relative [R|t] matrix
                rt_matrix_R = get_relative_rotation_translation(ematrix, k, imgL_matches, imgR_matches, fmap=fmap, verbose=verbose)

                # Compare this relative [R|t] matrix with the last one
                acceptable_z = check_rt_matrix(rt_matrix_R, verbose=verbose)
            
            current_trial += 1
            if verbose and current_trial < TRIALS and not acceptable_z:
                print("Starting trial {}, cap: {}".format(current_trial + 1, TRIALS))
        
        if not acceptable_z:
            # Skip this image
            print("Image skipped after {} trials".format(TRIALS))
            continue
        
        print('Trials required: {}'.format(current_trial))

        # Convert the relative rtmatrix above into a global rtmatrix
        global_rt_matrix_R = get_global_rotation_translation(global_rt_list[-1, :, :], rt_matrix_R, verbose=verbose)

        # Add the new global [R|t] matrix to the list
        global_rt_list = np.concatenate((global_rt_list, global_rt_matrix_R.reshape(1, 3, 4)))

        # Triangulate the matched feature points
        pts4D = triangulate_feature_points(global_rt_list, imgL_matches, imgR_matches, k, fmap=fmap, verbose=verbose)

        if verbose_img:
            # Plot the point cloud of points from this image match
            plot_point_cloud(pts4D, global_rt_matrix_R.reshape(1, 3, 4), verbose=verbose)
        
        # Stop early for testing purposes
        if count >= 10:
            break

    # Plot the 3D points
    plot_point_cloud(all_pts4D, global_rt_list, pts4D_indices=pts4D_indices, verbose=verbose)


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
        self.verbose=verbose

    def next(self):
        """Loads the next image from the given directory

        The images are loaded in alphabetical order, one at a time.
        """
        # Check if the index is out of range
        if self.index >= self.count:
            return False
        img = cv2.imread(self.path + self.images[self.index])
        self.index += 1
        return img
    
    def load(self, img_name):
        """Loads the image with the specified name
        
        Causes an error if there is no file with that name in the directory.
        """
        img = cv2.imread(self.path + img_name)
        return img
    
    def reset(self, new_index=0):
        """Resets the index counter

        Sets the index counter to the given value.

        :param new_index: The index to reset to. Default 0.
        """
        self.index = new_index

def get_feature_points(img, verbose=False, verbose_img=False, image_name='Img'):
    """Extract feature points

    Extracts SURF feature points from the given image.

    :param img: the image from which to extract feature points
    :param verbose_img: as in pipeline()
    :param image_name: the name of the image to be displayed
    """
    # Create the SURF feature detector - higher value = fewer points
    detector = cv2.xfeatures2d.SURF_create(750) #HYPERPARAM - 350, 750
    # Find the keypoints and descriptors in the image
    kp, desc = detector.detectAndCompute(img, None)

    if verbose:
        # Print the number of features found
        print("{} features detected".format(len(kp)))

    if verbose_img:
        # Display the image with its feature points
        disp_img = np.zeros_like(img)
        cv2.drawKeypoints(img, kp, disp_img)
        # Shrink the image so it can be displayed in a sensical way
        disp_img = downsize_img(disp_img)
        cv2.imshow(image_name, disp_img)
        if cv2.waitKey(0) == 113:
            cv2.destroyWindow(image_name)
    # Return the keypoints and descriptors
    return kp, desc

def match_feature_points(imgL, ptsL, descL, imgR, ptsR, descR, verbose=False, verbose_img=False, image_name='Feature point matches'):
    """Match feature points in the two images

    This should only be called on consecutive images, or correct
    matches are unlikely (or impossible) to be found.

    :param imgL: the first image
    :param ptsL: feature points from the first image (should be the
        first image read)
    :param descL: the feature point descriptors of points in ptsL
    :param imgR: the second image
    :param ptsR: feature points from the second image
    :param descR: the feature point descriptors of points in ptsR
    :param verbose_img: as in pipeline()
    :param image_name: the name of the image to be displayed
    """
    # Create a feature point matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Find the top 2 matches for each feature point
    matches = matcher.knnMatch(descL, descR, k=2)
    pre_filtering_matches = len(matches)
    # Apply the ratio test: x is best match, y is second best match, lower distance is better
    matches = [x for x, y in matches if x.distance < 0.7*y.distance] #HYPERPARAM - 0.5*distance

    if verbose:
        # Print the number of feature point matches found (before and after filtering)
        print("Feature matches before ratio test: {}\nFeature matches after ratio test: {}".format(pre_filtering_matches, len(matches)))

    if verbose_img:
        # Display the feature point matches between the two images
        disp_img = np.hstack((imgL, imgR))
        cv2.drawMatches(imgL, ptsL, imgR, ptsR, matches, disp_img)
        disp_img = downsize_img(disp_img)
        cv2.imshow(image_name, disp_img)
        if cv2.waitKey(0) == 113:
            cv2.destroyWindow(image_name)
    # Extract the matched imgL points and imgR points from the matches list
    matched_ptsL = np.array([ptsL[match.queryIdx].pt for match in matches])
    matched_ptsR = np.array([ptsR[match.trainIdx].pt for match in matches])
    # Return the list of matches, and list of matched points in img 1 and 2
    return matches, matched_ptsL, matched_ptsR

def get_fundamental_matrix(matched_ptsL, matched_ptsR, verbose=False):
    """Estimate the fundamental matrix

    Note: this is only valid for this pair of images

    :param pt_matches: the list of feature point matches. These should
        be from consecutive images
    :param matched_ptsL: the feature points in the first image
    :param matched_ptsR: the feature points in the second image
    :param verbose: as in pipeline()
    """
    # Randomly sort the matched pts lists to ensure this function yields a different output if repeated on the same input
    indices = [x for x in range(len(matched_ptsL))]
    # Shuffle the indices list
    random.shuffle(indices)
    # Now reconstruct the matched pts lists
    matched_ptsL = np.array([matched_ptsL[i, :] for i in indices])
    matched_ptsR = np.array([matched_ptsR[i, :] for i in indices])
    fmatrix, fmap = cv2.findFundamentalMat(matched_ptsL, matched_ptsR, cv2.FM_RANSAC, 0.3, 0.999) #HYPERPARAM - 0.1, 0.999
    if verbose:
        # Print the fundamental matrix
        print("Fundamental matrix:")
        with np.printoptions(suppress=True):
            print(fmatrix)
        # Print the count of inliers and outliers
        print("Inliers: {}\nOutliers: {}".format((fmap == 1).sum(), (fmap == 0).sum()))
    # Return the fundamental matrix
    return fmatrix, fmap

def get_essential_matrix(fmatrix, k, verbose=False):
    """Compute the essential matrix

    :param fmatrix: the fundamental matrix corresponding to this
        essential matrix
    :param k: the camera intrinsics relevant to the two images
        (it is assumed that this will be the same)
    :param verbose: as in pipeline()
    """
    # Essential matrix is calculated as E=K_1.T * F * K_2
    # Since the same camera is used for the entire video sequence, K_1 = K_2 = k
    ematrix = np.matmul(np.matmul(np.transpose(k), fmatrix), k)
    if verbose:
        # print the essential matrix
        print("Essential matrix:")
        with np.printoptions(suppress=True):
            print(ematrix)
    # Return the essential matrix
    return ematrix

def get_relative_rotation_translation(ematrix, k, ptsL, ptsR, fmap=None, verbose=False):
    """Compute the [R|t] matrix using cv2.recoverPose

    :param ematrix: the essential matrix computed from ptsL and ptsR
    :param k: the camera intrinsics matrix of the camera used to capture
        both image 1 and image 2
    :param ptsL: the feature points in image 1 with a match in image 2
    :param ptsR: the feature poitns in image 2 corresponding to ptsL
    :param fmap: A binary list denoting which points in ptsL and ptsR
        were inliers for the creation of the fundamental and essential
        matrices
    :param verbose: as in pipeline()
    """
    # Use recoverPose to compute the R and t matrices
    points, R, t, mask = cv2.recoverPose(ematrix, ptsL, ptsR, k, mask=fmap)
    #TODO: try to fix plot being upside down - can probably he HACK'ed here
    
    # Create the [R|t] matrix by stacking R and t horizontally
    rt_matrix = np.hstack((R, t))

    if verbose:
        # Print the [R|t] matrix
        print("[R|t] matrix:")
        with np.printoptions(suppress=True):
            print(rt_matrix)

    # Return the [R|t] matrix
    return rt_matrix

def check_rt_matrix(current_rt_matrix, verbose=False):
    """Check if the current [R|t] matrix is acceptable_z
    :param current_rt_matrix: the [R|t] matrix to be checked
    :param verbose: as in pipeline()
    """
    # Since the camera is mounted on a car, looking at 90' right or 90' left, there should be VERY little z translation
    # Thus, only allow -0.05 < z < 0.05 - even when the car is turning, this should work as the relative z translation
    #   from frame-to-frame will be very small
    current_z_translation = current_rt_matrix[2, 3]
    if current_z_translation < -0.05 or current_z_translation > 0.05:
        return False
    return True

def get_global_rotation_translation(global_rt_matrix_L, rt_matrix_R, verbose=False):
    """Compute the [R|t] matrix for imgR/camR from the first img/cam

    :param global_rt_matrix_L: The global [R|t] matrices corresponding to the
        previous image
    :param rt_matrix_R: The [R|t] matrix corresponding to the next image, to
        be converted into the global equivalent
    :param verbose: as in pipeline()
    """
    # Decompose the [R|t] matrices into R, t
    r_mat_L = global_rt_matrix_L[:, :3]
    t_L = global_rt_matrix_L[:, 3]
    r_mat_R = rt_matrix_R[:, :3]
    t_R = rt_matrix_R[:, 3]

    # Convert the rotation matrices to rotation vectors
    r_vec_L, _ = cv2.Rodrigues(r_mat_L)
    r_vec_R, _ = cv2.Rodrigues(r_mat_R)

    # Compose the new R and t vectors with the latest global R and t vectors (r_mat_L, t_L)
    global_r_vec_R, global_t_R, _, _, _, _, _, _, _, _ = cv2.composeRT(r_vec_L, t_L, r_vec_R, t_R)

    # Convert the R vector back into a matrix
    global_r_mat_R, _ = cv2.Rodrigues(global_r_vec_R)

    # Create the global [R|t] matrix
    global_rt_R = np.hstack((global_r_mat_R, global_t_R))

    if verbose:
        # Print the global_rt_R matrix
        print("Global [R|t] matrix:")
        with np.printoptions(suppress=True):
            print(global_rt_R)

    # Return the global_rt_R
    return global_rt_R

def triangulate_feature_points(global_rt_list, ptsL, ptsR, k, fmap=[], verbose=False):
    """Get 4D (homogeneous) points through triangulation

    :param global_rt_list: A list of all the global [R|t] matrices for all
        images processed so far, in order
    :param ptsL: the matched points in the left image
    :param ptsR: the matched points in the right image
    :param k: the camera intrinsics matrix
    :param fmap: a binary list mapping which matched points are inliers for
        the fundamental matrix
    :param verbose: as in pipeline()
    """
    # triangulate points
    # Multiply [R|t] by k to yield the camera perspective matrix
    r_L = global_rt_list[-2, :, :]
    r_L = np.matmul(k, r_L)
    r_R = global_rt_list[-1, :, :]
    r_R = np.matmul(k, r_R)

    # If fmap was provided, filter out non-inliers
    if len(fmap) > 0:
        # Only triangulate points which were inliers for the fundamental matrix
        inlier_pts_L = []; inlier_pts_R = []
        for i in range(len(fmap)):
            if fmap[i]:
                inlier_pts_L.append(ptsL[i,:])
                inlier_pts_R.append(ptsR[i,:])
        
        triang_pts_L = np.array(inlier_pts_L)
        triang_pts_R = np.array(inlier_pts_R)
    else:
        triang_pts_L = ptsL
        triang_pts_R = ptsR

    pts4D = cv2.triangulatePoints(r_L, r_R, triang_pts_L.T, triang_pts_R.T).T

    if verbose:
        # Print out how many points were triangulated
        print("{} points triangulated".format(pts4D.shape[0]))

    # Return the list of 4D (homogeneous) points
    return pts4D

def plot_point_cloud(pts4D, global_rt_list, pts4D_indices=[], verbose=False):
    """Create a point cloud of all 3D points found so far

    :param pts4D: a list of 4D (homogeneous) points to be plotted
    :param global_rt_list: a list of the global [R|t] matrices, from
        which the camera position for each image can be reconstructed
    :param pts4D_indices: a list of indices, separating the pts4D
        into one set for each pair of images processed. If this
        is the empty list, all points are plotted with the same
        colour.
    :param verbose: as in pipeline()
    """
    # convert from homogeneous coordinates to 3D
    pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

    # print(len(pts3D))
    # pts3D = np.array([point for point in pts3D if point[0] < 20 and point[1] < 20 and point[2] < 20])
    # print("Post processing:", len(pts3D))

    # print(pts3D.shape)
    # # Test the resulting Z coordinates
    # test_x = sorted(pts3D[:][0])
    # test_y = sorted(pts3D[:][1])
    # test_z = sorted(pts3D[:][2])

    # print("Lowest X:", test_x[0])
    # print("Second highest X:", test_x[-2])
    # print("Highest X:", test_x[-1])
    # print("Lowest Y:", test_y[0])
    # print("Second highest Y:", test_y[-2])
    # print("Highest Y:", test_y[-1])
    # print("Lowest Z:", test_z[0])
    # print("Second highest Z:", test_z[-2])
    # print("Highest z:", test_z[-1])

    colours = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 0.5, 0],
        [1, 0, 0.5],
        [0.5, 1, 0],
        [0.5, 0, 1],
        [0, 1, 0.5],
        [0, 0.5, 1],
        [1, 0.5, 0.5],
        [0.5, 1, 0.5],
        [0.5, 0.5, 1],
        [0, 0, 0]
    ]
    # cv2.projectPoints()
    # pcd.colors = Vector3dVector(np_colors)

    # Plot with open3d
    plot_list = []
    if len(pts4D_indices) == 0:
        # Plot all points red
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pts3D)
        pcd.paint_uniform_color([1, 0, 0])
        plot_list.append(pcd)
    else:
        # Plot each set of points (for each image pair) a different colour
        for index in range(1, len(pts4D_indices)):
            start = pts4D_indices[index - 1]
            end = pts4D_indices[index]
            pcd = open3d.PointCloud()
            pcd.points = open3d.Vector3dVector(pts3D[start:end])
            pcd.paint_uniform_color(colours[(index-1) % len(colours)])
            plot_list.append(pcd)
    
    #TODO: fix this or remove it, currently not working perfectly and is not a priority
    # Plot a cone at each camera position
    # for index, rt_matrix in enumerate(global_rt_list):
    #     # Create a small cone
    #     cam_cone = open3d.create_mesh_cone(radius=0.5, height=1.0)
    #     # Set the cone's colour to red
    #     cam_cone.paint_uniform_color(colours[(index) % len(colours)])
    #     # First flip the cone along the z axis
    #     cam_cone.transform(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]).reshape(4, 4))
    #     # Now transform the cone according to the (homogenized) [R|t] matrix
    #     # cam_cone.transform(np.vstack((rt_matrix.itemset(4, -rt_matrix.item(4)), np.array([0, 0, 0, 1]))))
    #     # rt_temp = np.copy(rt_matrix) #FIXME
    #     # rt_temp[0, 3] = -rt_temp[0, 3]
    #     # rt_temp[1, 3] = -rt_temp[1, 3]

    #     # cam_cone.transform(np.vstack((rt_temp, np.array([0, 0, 0, 1]))))
    #     # Alternative camera center computation
        

    #     # Add the cone to the list of things to plot
    #     plot_list.append(cam_cone)
    
    # Print the number of 3D points plotted
    print("Plotted {} 3D points".format(len(pts3D)))
    
    # Draw the point cloud
    open3d.draw_geometries(plot_list)

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
    :param wmax: the maximum acceptable_z width
    :param hmax: the maximum acceptable_z height
    """
    # Downsize img until its width is less than wmax and height is less than hmax
    while img.shape[1] > wmax or img.shape[0] > hmax:
        img = cv2.resize(img, None, fx=0.75, fy=0.75)
    return img
