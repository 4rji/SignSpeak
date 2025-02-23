import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model("asl_mnist_model.h5")

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))  # Resize to 28x28
    img = img.reshape(1, 28, 28, 1) / 255.0  # Normalize
    return img

# Predict ASL gesture
def predict_asl(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)  # Get predicted class
    letters = "ABCDEFGHIKLMNOPQRSTUVWXY"  # ASL letters (J & Z excluded)
    return letters[class_index]

# Test with a sample image
image_path = "dataset/sample_asl.jpg"  # Replace with an actual image path
predicted_letter = predict_asl(image_path)
print(f"Predicted ASL Letter: {predicted_letter}")
