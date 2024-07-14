"""
This script detects markers using Aruco from the webcam and draw pose
"""

# Import required packages
import cv2
import os
import pickle
import numpy as np
# Check for camera calibration data
path_to_calibraiton = "calibration/calibration.pkl"
if not os.path.exists('./' + path_to_calibraiton):
    print("You need to calibrate the camera you'll be using. See calibration script.")
    exit()
else:
    f = open(path_to_calibraiton, 'rb')
    calibration_data = pickle.load(f)
    # Extract the calibration data
    ret = calibration_data['ret']
    cameraMatrix = np.array(calibration_data['mtx'])  # Convert list back to NumPy array
    distCoeffs = np.array(calibration_data['dist'])  # Convert list back to NumPy array
    rvecs = calibration_data['rvecs']
    tvecs = calibration_data['tvecs']

    # Print the extracted data (optional)
    print("ret:", ret)
    print("mtx:", cameraMatrix)
    print("dist:", distCoeffs)
    print("rvecs:", rvecs)
    print("tvecs:", tvecs)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera")
        exit()
print("CameraMatrix",cameraMatrix)
print("DistCoeffs",distCoeffs)
# We create the dictionary object using predefined dictionary function
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

# Create parameters to be used when detecting markers:
parameters = cv2.aruco.DetectorParameters()

# Create video capture object 'capture' to be used to capture frames from the first connected camera:
capture = cv2.VideoCapture(0)

while True:
    # Capture frame by frame from the video capture object 'capture':
    ret, frame = capture.read()
    cv2.imshow('original', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # We convert the frame to grayscale:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # We call the function to detect markers
    corners, ids, rejectedImgPoints = cv2.aruco.ArucoDetector(aruco_dictionary).detectMarkers(gray_frame)

    # Draw detected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # Draw rejected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))

    if ids is not None:
    # Estimate pose for each marker
        #print(ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font for text display
        font_scale = 1  # Adjust font size as needed

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # Print marker ID on the frame
            text_str = f"ID: {ids[i]}"  # Create text string with ID
            (text_width, text_height) = cv2.getTextSize(text_str, font, font_scale, 2)[0]  # Get text size for positioning
            text_offset_x = 5  # Adjust X offset for better placement (optional)
            text_offset_y = corners[i][0][1] - text_height  # Place text above the top marker corner

            text_offset_y_int = int(text_offset_y[1])  # Convert to integer

            cv2.putText(frame, text_str, (text_offset_x, text_offset_y_int), font, font_scale, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw axis (optional)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, length=1)
    

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release everything:
capture.release()
cv2.destroyAllWindows()
