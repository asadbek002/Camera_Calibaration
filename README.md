# Camera_Calibaration
This is a Python code for camera calibration and pose estimation using OpenCV. The code uses a chessboard pattern of size 10x6 and a square size of 2.5 cm to find the corners of the chessboard in a series of images. The chessboard corner coordinates in the 3D real world space are saved in objpoints, while the 2D image points are saved in imgpoints. The camera is calibrated using the cv.calibrateCamera() function, which returns the camera matrix K, distortion coefficients dist, and rotation and translation vectors rvecs and tvecs.

After the camera is calibrated, the code estimates the pose of the camera using the cv.solvePnP() function, which returns the rotation and translation vectors rvecs and tvecs. The code also draws a coordinate axis on the image to visualize the camera's orientation in space.

# RESULT
<img src="https://github.com/asadbek002/Camera_Calibaration/blob/master/result_screenshot.jpg" width="500" height="300">

<img src="https://github.com/asadbek002/Camera_Calibaration/blob/master/.idea/video_screenshot.jpg" width="500" height="300">

