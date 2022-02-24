import numpy as np
import cv2 as cv
import glob

# Chessboard Size and Frame Size setting
chessboardSize = (23,23)
frameSize = (1758,989)

# Termination Criteria and Object Points
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

# Chessboard Square Size Setting
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Empty Arrays to save 3D and 2D Points
objpoints = [] # 3d point in World Coordinates
imgpoints = [] # 2d points in image(pixel) Coordinates

# Import Images(* means bringing all the images in the directory)
images = glob.glob('*.jpg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)
cv.destroyAllWindows()

print(corners)
print(ret)

# Calibration Using Object Points and Image Points
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print(dist) # Distortion
print(cameraMatrix) # Camera Matrix


# Position Calculation
# 3D Points Array
points_3D =np.array([
 
                      (0.0, 0.0, 0.0),

                      (20,0,0),
 
                      (0, 27, 0),
 
                      (20,27, 0),
 
                      (9,7,0),
 
                      (16,1,0) 
])
# 2D Points array
points_2D = np.array([
                        (382,969),
 
                        (705,973),
 
                        (367,542),
 
                        (695,533),
 
                        (521, 865),
 
                        (639,960)
 
                      ], dtype="double")
 

# SolvePnP to get Rotation and Translation Vector
success, rotation_vector, translation_vector = cv.solvePnP(points_3D, points_2D, cameraMatrix, dist, flags=0)

# Using cv.Rodrigues to get Rotation Matrix from Rotation Vector
rotation_Matrix = cv.Rodrigues(rotation_vector)[0]
print(rotation_Matrix)

# Calculatie Camera Position from Rotation Matrix and Translation Vector
cameraPosition = -np.matrix(rotation_Matrix).T * np.matrix(translation_vector)
print(cameraPosition)
