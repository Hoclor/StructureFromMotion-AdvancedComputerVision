# Original file copied from https://github.com/mbeyeler/opencv-python-blueprints/tree/master/chapter4
# and edited to fit the task of the assignment

import numpy as np

from pipeline import Pipeline

# Path to the directory containing the images to be used as input. The images should be present directly in this directory (i.e. not in folders inside it)
master_path_to_dataset = 'C:/Users/simon/GitRepositories/SSAIV_ACV/Dataset/2015-03-27_10-47-08_Seq2/monocular_left_calibrated/'

def main():
    # K and d of the left and right camera used for the 3 data sequences are known, if one of these is selected as the master_path_to_dataset load the appropriate K from K_array.

    K_left = np.array([1.2432403472640376e+003, 0., 6.6606587228414776e+002, 0.,
       1.2331555645540684e+003, 4.5437658085060713e+002, 0., 0., 1.]).reshape(3, 3)
    d_left = np.array([-4.6261205649774895e-001, 2.0732672793966445e-001,
       6.2375723417721863e-003, 2.6299185722505536e-003, -3.6689019973370124e-002]).reshape(1, 5)

    K_right = np.array([1.1930484416274905e+003, 0., 6.1463232877569749e+002, 0.,
       1.1864252309660656e+003, 4.3318083737428361e+002, 0., 0., 1.]).reshape(3, 3)
    d_right = np.array([-4.8441890104482732e-001, 3.1770182182461387e-001,
       4.8167296939537890e-003, 5.9334794668205733e-004, -1.4902486951308128e-001]).reshape(1, 5)

    # Check if the dataset is one of the 3 sequences, and if so if it's the right or left camera
    if '2015-03-27_10-47-08_' in master_path_to_dataset:
        if 'monocular_left_calibrated' in master_path_to_dataset:
            K = np.copy(K_left)
            d = np.copy(d_left)
        elif 'monocular_right_calibrated' in master_path_to_dataset:
            K = np.copy(K_right)
            d = np.copy(d_right)
        else:
            print('Error: Original data sequence identified, but neither right nor left camera chosen: {}'.format(master_path_to_dataset))
            K = np.zeros((3, 3))
            d = np.zeroes((1, 5))
    else:
        # Unexpected dataset: make the user confirm the images are in chronological order, and that they've inserted the values for K and d here
        #HERE
        K = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(3, 3)
        d = np.array([0,1,2,3,4]).reshape(1, 5)
        cont = input("A different dataset than the one given with the assignment was detected.\nPlease ensure that the images in the directory are in chronological order, alphabetically (i.e. first image alphabetically is also first image chronologically)\nAlso ensure that you have inserted the correct camera intrinsics values into K and d in the code, after the comment '#HERE'.\nContinue? (y/n)")
    if('y' != cont.lower() and 'yes' != cont.lower()):
        quit()


    # Create an instance of the Pipeline class
    pipeline_instance = Pipeline(K, d)
    # Execute the SfM pipeline on this image sequence

    # load a pair of images for which to perform SfM
    pipeline_instance.load_image_pair(master_path_to_dataset + "Images_cam0_7319.png", master_path_to_dataset + "Images_cam0_7320.png")

    # draw 3D point cloud of fountain
    # use "pan axes" button in pyplot to inspect the cloud (rotate and zoom
    # to convince you of the result)
    pipeline_instance.plot_point_cloud()


if __name__ == '__main__':
    main()