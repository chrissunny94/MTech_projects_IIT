import numpy as np
import cv2 as cv
import cv2
import glob
import numpy as np
import pickle
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)
image_count=0
while image_count<7:
    ret, frame = cap.read()
    print("Image_count", image_count)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
    file_path = "temp/"+str(image_count)+".jpg"
    print("writing to ",file_path)
    cv2.imwrite(file_path, frame)
    image_count= image_count+1

 
images = glob.glob('temp/*.jpg')
 
for fname in images:
 img = cv.imread(fname)
 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
 # Find the chess board corners
 ret, corners = cv.findChessboardCorners(gray, (9,6), None)
 
 # If found, add object points, image points (after refining them)
 if ret == True:
    objpoints.append(objp)
    
 
 corners2 = cv.cornerSubPix(gray,corners, (9,6), (-1,-1), criteria)
 imgpoints.append(corners2)
 
 # Draw and display the corners
 cv.drawChessboardCorners(img, (9,6), corners2, ret)
 cv.imshow('img', img)
 cv.waitKey(500)
 
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(objpoints)):
 imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
 error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
 mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )


# Create a dictionary to store the calibration data in a structured manner
calibration_data = {
    'ret': ret,
    'mtx': mtx.tolist(),  # Convert NumPy array to list for pickling
    'dist': dist.tolist(),  # Convert NumPy array to list for pickling
    'rvecs': rvecs,  # These can be lists or NumPy arrays (depending on structure)
    'tvecs': tvecs   # These can be lists or NumPy arrays (depending on structure)
}

# Open the pickle file in write binary mode ('wb')
with open('calibration/calibration.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)

print("Calibration data saved to calibration.pkl")