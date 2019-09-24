# Structure from Motion using Advanced Computer Vision
Code for the Structure from Motion coursework for the `Advanced Computer Vision` submodule as part of the `Software, Systems and Applications IV` module, taken in my fourth year of Computer Science (MEng) at Durham University.

## Requirements

The following Python libraries are required to run the pipeline:
- python (3.6.7 64-bit preferably)
- numpy
- opencv2 (version 3.3.*, or enable use of SURF when building the library)
- open3d

## Structure from Motion pipeline

For each neighbouring image pair (i.e. 0 and 1, 1 and 2, ...) the following steps are completed to extract 3D structure from the motion of the camera:
1. Feature points are extracted
2. Feature point are matched between the images
3. The Fundamental matrix is estimated through a RANSAC approach on the feature matches
4. The Essential matrix is estimated from the Fundamental matrix using least squares optimisation
5. The relative camera matrix (rotation and translation) of the second camera from the first is computed from the Essential matrix
6. The previous and current camera matrices are composed to produce global camera matrix of the first and second camera
7. Matched feature points are triangulated using the global camera matrix  to produce 3D points
8. A 3D point cloud is creation and plotted for visualization of the 3D structure
