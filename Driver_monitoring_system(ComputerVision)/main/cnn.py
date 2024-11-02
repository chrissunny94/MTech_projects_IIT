import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_cnn(input_shape, num_classes):
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional layer
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output of the convolutional layers to feed into dense layers
    model.add(Flatten())
    
    # First dense (hidden) layer
    model.add(Dense(128, activation='relu'))
    
    # Second dense (hidden) layer
    model.add(Dense(64, activation='relu'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Example usage
input_shape = (68, 68, 3)
num_classes = 10  # Change this according to your number of classes

model = create_cnn(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
