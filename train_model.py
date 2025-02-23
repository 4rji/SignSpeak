import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the dataset
train_data = pd.read_csv("dataset/sign_mnist_train.csv")
test_data = pd.read_csv("dataset/sign_mnist_test.csv")

# Separate labels and features
y_train = train_data["label"].values
y_test = test_data["label"].values
X_train = train_data.drop("label", axis=1).values
X_test = test_data.drop("label", axis=1).values

# Reshape images (28x28 pixels)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0  # Normalize to 0-1 range
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=25)  # ASL A-Z (except J & Z)
y_test = to_categorical(y_test, num_classes=25)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='softmax')  # 25 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("asl_mnist_model.h5")
