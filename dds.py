import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Define parameters
img_width, img_height = 64, 64
batch_size = 32
epochs = 20
dataset_dir = 'mrl_dataset'  # Update this to your dataset path

# Step 2: Load dataset
def load_data(dataset_dir):
    images = []
    labels = []
    
    # Loop through Drowsy and Normal folders
    for label in ['Normal', 'Drowsy']:
        dir_path = os.path.join(dataset_dir, label)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(0 if label == 'Normal' else 1)  # Normal = 0, Drowsy = 1
    
    return np.array(images), np.array(labels)

# Load the images and labels
images, labels = load_data(dataset_dir)
images = images.astype('float32') / 255.0  # Normalize pixel values
labels = np.array(labels)

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 4: Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.1,
                             zoom_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

# Step 5: Define CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Step 6: Compile model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), 
          validation_data=(X_test, y_test), 
          epochs=epochs)

# Step 8: Save the model
model.save('drowsiness_detection_model.h5')

# Step 9: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Real-Time Drowsiness Detection
def real_time_detection():
    # Load the trained model
    model = load_model('drowsiness_detection_model.h5')

    # Start video capture from the webcam
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        img = cv2.resize(frame, (64, 64))  # Resize to match model input
        img = img.astype('float32') / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make a prediction
        prediction = model.predict(img)
        predicted_class = np.round(prediction).astype(int)[0][0]

        # Determine the label
        label = 'Drowsy' if predicted_class == 1 else 'Normal'
        
        # Display the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Drowsiness Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Uncomment the following line to run real-time detection after training
# real_time_detection()