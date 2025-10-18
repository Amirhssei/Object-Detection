# Import necessary libraries
import tensorflow as tf
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import cv2
import numpy as np

# ---------------------------------------------------------
# Load CIFAR-10 dataset (contains 60,000 32x32 color images in 10 categories)
# ---------------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values (0–255 → 0–1) for faster convergence
train_images, test_images = train_images / 255.0, test_images / 255.0

print("Training data shape:", train_images.shape)
print("Test data shape:", test_images.shape)

# Define class names for better readability
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# ---------------------------------------------------------
# Build the Convolutional Neural Network (CNN)
# ---------------------------------------------------------
model = Sequential([
    # First convolution layer with 32 filters and ReLU activation
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    # Second convolution layer with 64 filters
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Third convolution layer
    Conv2D(64, (3, 3), activation='relu'),

    # Flatten and feed into dense layers
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10)  # Output layer with 10 classes
])

# Display model structure
model.summary()

# ---------------------------------------------------------
# Compile the model with Adam optimizer and cross-entropy loss
# ---------------------------------------------------------
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ---------------------------------------------------------
# Train the model for 10 epochs and validate with test data
# ---------------------------------------------------------
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# ---------------------------------------------------------
# Test the model on test dataset
# ---------------------------------------------------------
predictions = model.predict(test_images)

# Example: Show prediction for the first image in test set
predicted_class_index = np.argmax(predictions[0])
print("Predicted class:", class_names[predicted_class_index])

# ---------------------------------------------------------
# Function to predict custom image class
# ---------------------------------------------------------
def predict_custom_image(model, image_path):
    """Reads, preprocesses, and predicts the class of a given image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))  # Resize to match model input
    img = img / 255.0                # Normalize
    img = np.array([img])            # Convert to batch format

    prediction = model.predict(img)
    predicted_index = np.argmax(prediction[0])
    print("Predicted class for", image_path, ":", class_names[predicted_index])

# ---------------------------------------------------------
# Predict custom images
# ---------------------------------------------------------
predict_custom_image(model, 'horse.jpg')
predict_custom_image(model, 'bird.jpg')

# ---------------------------------------------------------
# Save and reload the trained model
# ---------------------------------------------------------
model.save('model.h5')
model.load_weights('cifar10_cnn_model.weights.h5')
