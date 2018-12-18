# SSAIV_ACV
Code for Structure from Motion assignment for Advanced Computer Vision as part of Software, Systems and Applications IV module, taken in my fourth year of Computer Science at Durham University.

## Requirements

The following Python libraries are required to run the pipeline:
- python (3.6.7 64-bit preferably)
- numpy
- opencv2 (version 3.3.*, or enable use of SURF when building the library)
- open3d

## Structure from Motion pipeline

For each neighbouring image pair (i.e. 0 and 1, 1 and 2, ...)
- Feature point extraction
- Feature point matching between the images
- Estimation of the Fundamental matrix through RANSAC on the feature matches
- Computation of the Essential matrix
- Computation of relative camera matrix (rotation and translation) of the second camera from the first
- Camera matrix composition to produce global camera matrix of the first and second camera
- Triangulation of matched feature points to produce 3D points
- Creation and plotting of 3D point cloud for visualization
