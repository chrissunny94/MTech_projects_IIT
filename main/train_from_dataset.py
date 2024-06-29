import dlib
import cv2
import numpy as np
import os
from imutils import face_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


path = "artifacts/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images



images= load_images_from_folder("/Users/christhaliyath/MTECH/dataset/archive/dms/image")

for image in images:
    # cv2.imshow("Output", image)
    # k = cv2.waitKey(0) & 0xFF
    # if k == 27:
    #     break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    height, width = gray.shape
    print(height, width)
    mask = np.zeros((height, width, 3), np.uint8)
    
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (X, Y) in shape:
            cv2.circle(image, (X, Y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    # cv2.imshow("Output", image)
    # show the output image with the face detections + facial landmarks
    cropped_output = image[y:y+h,x:x+w]
    print( cropped_output.shape)
    cv2.imshow("Dlib_output", cropped_output)
    gray = cv2.cvtColor(cropped_output, cv2.COLOR_BGR2GRAY)
    width,height,channel = cropped_output.shape
    
    #DEEP LEARNING PART
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(width, height,channel)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    num_classes=2  # based on the dataset
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    batch_size = 2
    epochs = 1
    x_train = [cropped_output , cropped_output,cropped_output]
    y_train = ['closed','closed','closed']
    x_test = [cropped_output , cropped_output,cropped_output]
    y_test = ['closed','closed','closed']
    
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("test_model.h5")

    
cv2.waitKey(0)
#cv2.destroyAllWindows()



