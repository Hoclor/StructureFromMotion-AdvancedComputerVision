# SSAIV_ACV
Code for Structure from Motion assignment for Advanced Computer Vision as part of Software, Systems and Applications IV module, taken in my fourth year of Computer Science at Durham University.

Structure from Motion pipeline:
- Feature extraction
- Feature matching across subsequent images
- Extract perspective transforms (rotation and translation)
    - Fundamental matrix
    - Essential matrix
- Optimization of feature matches/perspective transform (RANSAC)
- 3D terrain point cloud