#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A module containing the pipeline used to achieve structure from motion to yield a point cloud """

import cv2
import numpy as np
import sys
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
    # load the first image into img2 (i.e. pretend we just processed this image in a previous pair)
    print('Processing image 1 out of {}'.format(img_loader.count))
    img2 = img_loader.next()
    #img2 = img_loader.load('Images_cam0_6517.png')

    # Extract feature points from this image
    pts2, desc2 = get_feature_points(img2, verbose_img=verbose_img)

    # Initialize some variables used for the entire sequence
    rtmatrix1 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=np.float64).reshape(3, 4) # The global [R|t] matrix for image 1
    global_Rt_list = rtmatrix1.reshape(1, 3, 4) # List of the global [R|t] matrices for each image
    all_pts4D = None # List of all 4D points, to be plotted at the end
    pts4D_indices = [0]
    # Loop over all the other images
    for count in range(1, img_loader.count):
        print('Processing image {} out of {}'.format(count+1, img_loader.count))
        # Move the last image (img2) into img1, and its points into pts1
        img1 = img2
        pts1, desc1 = pts2, desc2

        # Load the next image into img2
        img2 = img_loader.next()
        #img2 = img_loader.load('Images_cam0_6518.png')

        # Extract feature points of the second image
        pts2, desc2 = get_feature_points(img2, verbose_img=verbose_img)

        # Find the feature point matches between img1 and img2
        matches, img1_matches, img2_matches = match_feature_points(img1, pts1, desc1, img2, pts2, desc2, verbose_img=verbose_img)

        # Estimate the fundamental matrix
        fmatrix, fmap = get_fundamental_matrix(img1_matches, img2_matches, verbose=verbose)

        # Calculate the essential matrix
        ematrix = get_essential_matrix(fmatrix, k, verbose=verbose)

        # Get the relative [R|t] matrix
        rtmatrix2 = get_relative_rotation_translation(ematrix, k, img1_matches, img2_matches, fmap=fmap, verbose=verbose)

        # Convert the relative rtmatrix above into a global rtmatrix
        global_rtmatrix2 = get_global_rotation_translation(global_Rt_list[-1, :, :], rtmatrix2, verbose=verbose)

        # Add the new global [R|t] matrix to the list
        global_Rt_list = np.concatenate((global_Rt_list, global_rtmatrix2.reshape(1, 3, 4)))

        # Triangulate the matched feature points
        pts4D = triangulate_feature_points(global_Rt_list, img1_matches, img2_matches, k, verbose=verbose)

        # Add the list of 4D pts to the list of all 4D pts
        if type(all_pts4D) != type(None):
            pts4D_indices.append(len(all_pts4D))
            all_pts4D = np.concatenate((all_pts4D, pts4D))
        else:
            all_pts4D = pts4D

        # plot_point_cloud(pts4D, verbose=verbose)
        
        # Stop early for testing purposes
        if count >= 5:
            break

    # Plot the 3D points
            plot_point_cloud(all_pts4D, pts4D_indices=pts4D_indices, method='open3d', verbose=verbose)


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
        index = new_index

def get_feature_points(img, verbose_img=False, image_name='Img'):
    """Extract feature points

    Extracts SURF feature points from the given image.

    :param img: the image from which to extract feature points
    :param verbose_img: as in pipeline()
    :param image_name: the name of the image to be displayed
    """
    # Create the SURF feature detector
    detector = cv2.xfeatures2d.SURF_create(350)
    # Find the keypoints and descriptors in the image
    kp, desc = detector.detectAndCompute(img, None)

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

def match_feature_points(img1, pts1, desc1, img2, pts2, desc2, verbose_img=False, image_name='Feature point matches'):
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
    :param verbose_img: as in pipeline()
    :param image_name: the name of the image to be displayed
    """
    # Create a feature point matcher
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Find the top 2 matches for each feature point
    matches = matcher.knnMatch(desc1, desc2, k=2)
    # Apply the ratio test: x is best match, y is second best match, lower distance is better
    matches = [x for x, y in matches if x.distance < 0.5*y.distance]

    if verbose_img:
        # Display the feature point matches between the two images
        disp_img = np.hstack((img1, img2))
        cv2.drawMatches(img1, pts1, img2, pts2, matches, disp_img)
        disp_img = downsize_img(disp_img)
        cv2.imshow(image_name, disp_img)
        if cv2.waitKey(0) == 113:
            cv2.destroyWindow(image_name)
    # Extract the matched img1 points and img2 points from the matches list
    matchedPts1 = np.array([pts1[match.queryIdx].pt for match in matches])
    matchedPts2 = np.array([pts2[match.trainIdx].pt for match in matches])
    # Return the list of matches, and list of matched points in img 1 and 2
    return matches, matchedPts1, matchedPts2

def get_fundamental_matrix(matchedPts1, matchedPts2, verbose=False):
    """Estimate the fundamental matrix

    Note: this is only valid for this pair of images

    :param pt_matches: the list of feature point matches. These should
        be from consecutive images
    :param matchedPts1: the feature points in the first image
    :param matchedPts2: the feature points in the second image
    :param verbose: as in pipeline()
    """
    fmatrix, fmap = cv2.findFundamentalMat(matchedPts1, matchedPts2, cv2.FM_RANSAC, 0.1, 0.999)
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
    # Since the same camera is used for the entire video sequence, K_1 == K_2 == k
    ematrix = np.matmul(np.matmul(np.transpose(k), fmatrix), k)
    if verbose:
        # print the essential matrix
        print("Essential matrix:")
        with np.printoptions(suppress=True):
            print(ematrix)
    # Return the essential matrix
    return ematrix

def _get_relative_rotation_translation_OLD(ematrix, k, pts1, pts2, verbose=False):
    """Compute the [R|t] matrix for img2/cam2 from img1/cam1

    :param ematrix: the essential matrix corresponding to these two
        images/cameras
    :param pts1: the matched points in the first image
    :param pts2: the corresponding matched points in the second image
    :param verbose: as in pipeline()
    """
    def _in_front_of_both_cameras(first_points, second_points, rot,
                                    trans):
        """Determines whether point correspondences are in front of both
        images.
        Copied from https://github.com/mbeyeler/opencv-python-blueprints/tree/master/chapter4
        """
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0]*rot[2, :],
                                trans) / np.dot(rot[0, :] - second[0]*rot[2, :],
                                                second)
            first_3d_point = np.array([first[0] * first_z,
                                        second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                        trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True
    # Decompose the essential matrix through singular value decomposition to get U, s, V.T
    U, s, V = np.linalg.svd(ematrix)
    V = V.T # convert V.T to V

    # Construct sigma as [[s, 0, 0], [0, s, 0], [0, 0, 0]]
    # for the lowest value in s (list of possible s values, sorted so lowest value is at index -1)
    sigma = np.array([s[-1], 0, 0, 0, s[-1], 0, 0, 0, 0]).reshape(3, 3)

    # Construct W as [[0, -1, 0,], [1, 0, 0], [0, 0, 1]]. Note: W.T = W^(-1)
    W = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1]).reshape(3, 3)
    # W = W.T

    # Compute R = U * W^(-1) * V.T
    R = np.matmul(np.matmul(U, W.T), V.T)

    # # Compute [t]_x = U * W * sigma * U.T
    # t_x = np.matmul(np.matmul(np.matmul(U, W), sigma), U.T)
    # # Extract the values of t from t_x, as t_x = 
    #     #  0   -t_3  t_2
    #     #  t_3  0   -t_1
    #     # -t_2  t_1  0
    # t = np.array([t_x[2][1], t_x[0][2], t_x[1][0]]).reshape(3, 1)

    # Compute [t]_x using alternative definition of U * Z * U.T, where Z =
        #  0 1 0
        # -1 0 0
        #  0 0 0
    Z = np.array([0, 1, 0, -1, 0, 0, 0, 0, 0]).reshape(3, 3)
    t_x = np.matmul(np.matmul(U, Z), U.T)
    t = np.array([t_x[2][1], t_x[0][2], t_x[1][0]]).reshape(3, 1)

    # Check if this [R|t] matrix actually makes sense, i.e. the 3D points produced are in front of the camera
    # This must be done as there are four possible [R|t] matrices, but only one makes sense in practice
    # The four possibilities are constructed through setting W = W.T and/or t = -t in the computation above

    # Convert the input pts1 and pts2 to homogeneous coordinates
    pts1_h = []
    print("k:", k.shape)
    for point in pts1:
        pts1_h.append(np.linalg.inv(k).dot([point[0], point[1], 1.0]))
    pts2_h = []
    for point in pts2:
        pts2_h.append(np.linalg.inv(k).dot([point[0], point[1], 1.0]))
    if not _in_front_of_both_cameras(pts1_h, pts2_h, R, t):
        # Try t = -t
        t = np.negative(t)

        # Check again
        if not _in_front_of_both_cameras(pts1_h, pts2_h, R, t):
            # Try W = W.T and t = -t
            R = np.matmul(np.matmul(U, W), V.T)
            
            # Check again
            if not _in_front_of_both_cameras(pts1_h, pts2_h, R, t):
                # It must now be W = W.T and t = t
                t = np.negative(t)
    
    #HACK
    # Check if the diagonal of R is all negative values. If so, negate R
    if R[0, 0] < 0 and R[1, 1] < 0 and R[2, 2] < 0:
        R = np.negative(R)

    # Construct [R|t] matrix
    Rt = np.hstack((R, t))

    if verbose:
        # Print out the [R|t] matrix
        print("[R|t] matrix:")
        with np.printoptions(suppress=True):
            print(Rt)
    
    # Return the [R|t] matrix
    return Rt

def get_relative_rotation_translation(ematrix, k, pts1, pts2, fmap=None, verbose=False):
    """Compute the [R|t] matrix using cv2.recoverPose

    :param ematrix: the essential matrix computed from pts1 and pts2
    :param k: the camera intrinsics matrix of the camera used to capture
        both image 1 and image 2
    :param pts1: the feature points in image 1 with a match in image 2
    :param pts2: the feature poitns in image 2 corresponding to pts1
    :param fmap: A binary list denoting which points in pts1 and pts2
        were inliers for the creation of the fundamental and essential
        matrices
    :param verbose: as in pipeline()
    """    
    # Use recoverPose to compute the R and t matrices
    points, R, t, mask = cv2.recoverPose(ematrix, pts1, pts2, k, mask=fmap)
    
    # Create the [R|t] matrix by stacking R and t horizontally
    Rt = np.hstack((R, t))

    if verbose:
        # Print the [R|t] matrix
        print("[R|t] matrix:")
        with np.printoptions(suppress=True):
            print(Rt)

    # Return the [R|t] matrix
    return Rt

def get_global_rotation_translation(global_Rt_last, rtmatrix, verbose=False):
    """Compute the [R|t] matrix for img2/cam2 from the first img/cam

    :param global_Rt_last: The global [R|t] matrices corresponding to the
        previous image
    :param rtmatrix: The [R|t] matrix corresponding to the next image, to
        be converted into the global equivalent
    :param verbose: as in pipeline()
    """
    # Decompose the [R|t] matrices into R, t
    R_0 = global_Rt_last[:, :3]
    t_0 = global_Rt_last[:, 3]
    R_1 = rtmatrix[:, :3]
    t_1 = rtmatrix[:, 3]
    

    # Convert the rotation matrices to rotation vectors
    R_0vec, _ = cv2.Rodrigues(R_0)
    R_1vec, _ = cv2.Rodrigues(R_1)

    # Compose the new R and t vectors with the latest global R and t vectors (R_0, t_0)
    global_R_vec, global_t, _, _, _, _, _, _, _, _ = cv2.composeRT(R_0vec, t_0, R_1vec, t_1)

    # Convert the R vector back into a matrix
    global_R, _ = cv2.Rodrigues(global_R_vec)

    # Create the global [R|t] matrix
    global_Rt = np.hstack((global_R, global_t))

    if verbose:
        # Print the global_Rt matrix
        print("Global [R|t] matrix:")
        with np.printoptions(suppress=True):
            print(global_Rt)

    # Return the global_Rt
    return global_Rt

def triangulate_feature_points(global_Rt_list, pts1, pts2, k, verbose=False):
    """Get 4D (homogeneous) points through triangulation

    :param global_Rt_list: A list of all the global [R|t] matrices for all
        images processed so far, in order
    :param pt_matches: the list of feature point matches between img1 and img2
    :param verbose: as in pipeline()
    """
    # triangulate points
    # first_inliers = np.array(pts1).reshape(-1, 3)[:, :2]
    # second_inliers = np.array(pts2).reshape(-1, 3)[:, :2]
    r_1 = global_Rt_list[-2, :, :]
    r_1 = np.matmul(k, r_1)
    r_2 = global_Rt_list[-1, :, :]
    r_2 = np.matmul(k, r_2)

    pts4D = cv2.triangulatePoints(r_1, r_2, pts1.T, pts2.T).T

    if verbose:
        # Print out how many points were triangulated
        print("{} points triangulated".format(pts4D.shape[0]))

    # Return the list of 4D (homogeneous) points
    return pts4D

#TODO: if the above function only finds relative/local 3D position, this function is needed
# def adjust_feature_points():
#     """

#     """
#     pass

def plot_point_cloud(pts4D, pts4D_indices=[], method='open3d', verbose=False):
    """Create a point cloud of all 3D points found so far

    :param pts4D: a list of 4D (homogeneous) points to be plotted
    :param pts4D_indices: a list of indices, separating the pts4D
        into one set for each pair of images processed. If this
        is the empty list, all points are plotted with the same
        colour.
    :param method: the method used to plot the points,
        either 'open3d' or 'matplotlib'
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
        [0, 0, 0],
        [1, 1, 1]
    ]

    if(method == 'open3d'):
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

        # Create a set of axes centered at the origin
        axes = open3d.create_mesh_coordinate_frame(size = 3, origin = [0, 0, 0])
        plot_list.append(axes)

        # Draw the point cloud
        open3d.draw_geometries(plot_list)

    elif(method == 'matplotlib'):
        # Plot with matplotlib
        Xs = pts3D[:, 0] #Ys
        Ys = pts3D[:, 1] #Zs
        Zs = pts3D[:, 2] #Xs

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xs, Ys, Zs, c='r', marker='o')
        ax.set_xlabel('X') #Z
        ax.set_ylabel('Y') #X
        ax.set_zlabel('Z') #Y
        plt.title('3D point cloud: Use pan axes button below to inspect')
        plt.show()

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
