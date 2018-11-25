# SSAIV_ACV
Code for Structure from Motion assignment for Advanced Computer Vision as part of Software, Systems and Applications IV module, taken in my fourth year of Computer Science at Durham University.

Structure from Motion pipeline:
For each neighbouring image pair (i.e. 0+1, 1+2, 2+3, ...)
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
After the above
- Bundle adjustment to improve global positioning
- Create and visualise 3D terrain point cloud
    - matplotlib?
