# SSAIV_ACV
Code for Structure from Motion assignment for Advanced Computer Vision as part of Software, Systems and Applications IV module, taken in my fourth year of Computer Science at Durham University.

Required Python libraries:
- numpy
- opencv2 (version 3.3.*, or enable use of SURF)
- open3d

Structure from Motion pipeline:
For each neighbouring image pair (i.e. 0 and 1, 1 and 2, ...)
- Feature extraction in both images
- Feature matching between images
- Fundamental matrix estimation
    - RANSAC:
        - Pick 8 feature matches
        - Compute F
        - Find inliars with F amongst other feature matches
        - Pick F with most inliars
- Compute Essential matrix
    - From F, camera intrinsics
- Get camera perspective transform (rotation and translation)
    - Relative to previous camera and/or absolute change from 'zero' position (i.e. from first image in dataset)
- Triangulate feature points to get 3D points
- Modify the resulting 3D points with camera rotation and translation to yield correct global position
After the above
- Bundle adjustment to improve final result/global positioning
- Create and visualise 3D terrain point cloud
    - matplotlib?

Extra functionality:
- Apply dense stereo to yield a more dense point cloud/3D model