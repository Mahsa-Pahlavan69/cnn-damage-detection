
"""1. Mount Google Drive:

If you're using Google Colab and your data/models are stored in Google Drive, you need to mount the drive to access your files.
"""

# ============================
#      Mount Google Drive
# ============================

from google.colab import drive
drive.mount('/content/drive')

"""2. Import Libraries:

Import all necessary libraries for data handling, model building, evaluation, and visualization.
"""

# ============================
#     Import Libraries
# ============================

import numpy as np
import scipy.io as sio
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt

"""3. Define Parameters and Paths:

Set hyperparameters for the CNN, and define paths for data and model storage.
"""

# ============================
#         Parameters
# ============================

# CNN Hyperparameters
kernel_size = 41
sub_sampling_factor = 2
conv_neurons = [64, 32]
fc_neurons = 10
learning_rate = 0.001
epochs = 20
batch_size = 32

# Paths
data_dir = '/content/drive/MyDrive/processed_data/'    # Directory containing processed_data folders for each joint
models_dir = '/content/drive/MyDrive/cnn_models/'      # Directory to save trained CNN models
new_test_data_dir = '/content/drive/MyDrive/new_test_data/'  # Directory containing new test .mat files

# Create directories if they don't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

"""4. Define Functions:

    train_and_save_cnn: This function trains a CNN for a specific joint and saves the trained model.

    calculate_pod: This function computes the Probability of Damage (PoD) based on predictions.
"""

# ============================
#   Function Definitions
# ============================

def train_and_save_cnn(joint, data_dir, models_dir):
    """
    Train a CNN model for a specific joint and save the trained model.

    Parameters:
    - joint (int): Joint number (1-30)
    - data_dir (str): Directory containing processed data
    - models_dir (str): Directory to save trained models
    """
    print(f"\n--- Training CNN for Joint {joint} ---")

    joint_path = os.path.join(data_dir, f"Joint_{joint}")
    train_data_path = os.path.join(joint_path, 'train_data.mat')

    # Check if training data exists
    if not os.path.exists(train_data_path):
        print(f"Training data for Joint {joint} not found at {train_data_path}. Skipping...")
        return

    # Load training data
    train_data = sio.loadmat(train_data_path)
    X_train = train_data['train_data']
    y_train = train_data['train_labels'].flatten()

    # Preprocess data: Expand dimensions for Conv1D (samples, timesteps, channels)
    X_train = np.expand_dims(X_train, axis=-1)  # Shape: (num_samples, 128, 1)

    # Build the CNN model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))  # Input layer

    # Add convolutional and pooling layers
    for neurons in conv_neurons:
        model.add(Conv1D(neurons, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=sub_sampling_factor))

    # Add flatten and fully connected layers
    model.add(Flatten())
    model.add(Dense(fc_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1)

    # Save the trained model
    model_filename = f'cnn_model_joint{joint}.h5'
    model_save_path = os.path.join(models_dir, model_filename)
    model.save(model_save_path)
    print(f"Model for Joint {joint} saved at {model_save_path}.")

    # Optional: Plot training history
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Joint {joint} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Joint {joint} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def calculate_pod(y_pred):
    """
    Calculate Probability of Damage (PoD).

    Parameters:
    - y_pred (numpy.ndarray): Binary predictions (0 for undamaged, 1 for damaged)

    Returns:
    - PoD (float): Probability of Damage
    """
    D_i = np.sum(y_pred)  # Number of frames classified as damaged
    T_i = len(y_pred)     # Total number of frames
    PoD_i = D_i / T_i
    return PoD_i

"""5. Training Loop:

Iterate over each joint (1 to 30), train the CNN model using the training data, and save the trained model.

Key Steps in train_and_save_cnn:

    Load Training Data: Load train_data.mat for the current joint.
    Preprocess Data: Expand dimensions to match Conv1D input requirements (samples, timesteps, channels).
    Build CNN Model: Define a Sequential model with specified Conv1D, MaxPooling1D, Flatten, Dense, and Dropout layers.
    Compile Model: Use Adam optimizer with specified learning rate and binary cross-entropy loss.
    Train Model: Train the model with the training data, using 20% for validation.
    Save Model: Save the trained model as cnn_model_joint{joint}.h5.
    Optional Visualization: Plot training and validation accuracy and loss.
"""

# ============================
#      Training Loop
# ============================

# Iterate over each joint to train and save models
for joint in range(1, 31):
    train_and_save_cnn(joint, data_dir, models_dir)

"""6. PoD Calculation:

After training, iterate over each joint to calculate PoD using the corresponding trained model and new test data.

Key Steps:

    Load Trained Model: Load the CNN model specific to the joint.
    Load Test Data: Load the new test data (Joint_{joint}.mat).
    Preprocess Data: Expand dimensions to match Conv1D input requirements.
    Make Predictions: Predict probabilities and convert them to binary predictions using a threshold of 0.5.
    Calculate PoD: Compute PoD as the ratio of frames predicted as damaged to total frames.
    Optional Visualization: Plot predicted probabilities for the first 100 frames to visualize model performance.
"""

# ============================
#     PoD Calculation
# ============================

# Initialize list to store PoD for each joint
PoD = []

# Iterate over each joint to calculate PoD
for joint in range(1, 31):
    print(f"\n--- Calculating PoD for Joint {joint} ---")

    # Define paths
    model_path = os.path.join(models_dir, f'cnn_model_joint{joint}.h5')
    test_data_path = os.path.join(new_test_data_dir, f'Joint_{joint}.mat')

    # Check if model and test data exist
    if not os.path.exists(model_path):
        print(f"Model for Joint {joint} not found at {model_path}. Skipping PoD calculation for this joint.")
        continue
    if not os.path.exists(test_data_path):
        print(f"Test data for Joint {joint} not found at {test_data_path}. Skipping PoD calculation for this joint.")
        continue

    # Load the trained CNN model
    model = load_model(model_path)
    print(f"Loaded model for Joint {joint} from {model_path}.")

    # Load test data
    test_mat = sio.loadmat(test_data_path)
    if 'normalizedData' not in test_mat:
        print(f"'normalizedData' variable not found in {test_data_path}. Skipping PoD calculation for this joint.")
        continue
    X_test = test_mat['normalizedData']  # Shape: (2048, 128)

    # Preprocess data: Expand dimensions for Conv1D
    X_test = np.expand_dims(X_test, axis=-1)  # Shape: (2048, 128, 1)

    # Make predictions
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)  # Binary predictions

    # Calculate PoD
    PoD_i = calculate_pod(y_pred)
    PoD.append(PoD_i)
    print(f"Joint {joint}: PoD = {PoD_i:.2f}")

    # Optional: Visualize predictions (e.g., first 100 frames)
    plt.figure(figsize=(10, 4))
    plt.plot(y_pred_prob[:100], label='Predicted Probability')
    plt.axhline(0.5, color='r', linestyle='--', label='Threshold')
    plt.title(f'Joint {joint} - Prediction Probabilities (First 100 Frames)')
    plt.xlabel('Frame')
    plt.ylabel('Probability of Damage')
    plt.legend()
    plt.show()

"""7. PoD Summary and Visualization:

After processing all joints, summarize and visualize the PoD values.

Explanation:

    Print PoD Summary: Lists the PoD for each joint.
    Bar Chart Visualization: Provides a visual representation of PoD across all joints, making it easy to identify joints with high PoD values.
    Identify Damaged Joints: Lists joints with PoD exceeding a specified threshold (e.g., 0.5), indicating potential damage.
"""

# ============================
#      PoD Summary
# ============================

# Summary of PoD for all joints
print("\n--- Summary of Probability of Damage (PoD) for all Joints ---")
for joint, pod in enumerate(PoD, start=1):
    print(f"Joint {joint}: PoD = {pod:.2f}")

# Visualization: Bar chart of PoD per joint
plt.figure(figsize=(15, 6))
plt.bar(range(1, 31), PoD, color='skyblue')
plt.xlabel('Joint Number')
plt.ylabel('Probability of Damage (PoD)')
plt.title('Probability of Damage (PoD) per Joint')
plt.xticks(range(1, 31))
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Identify joints with high PoD (e.g., PoD > threshold)
threshold = 0.5  # Define your threshold based on validation and desired sensitivity
damaged_joints = [joint for joint, pod in enumerate(PoD, start=1) if pod > threshold]
print(f"\nJoints with PoD > {threshold}: {damaged_joints}")

"""8. Optional: Save PoD Results to a File

To keep a record of your PoD calculations, you can save the results to a text file.
"""

# ============================
#        Optional Saving
# ============================

# Save PoD results to a text file
results_save_path = os.path.join(new_test_data_dir, 'PoD_results.txt')
with open(results_save_path, 'w') as f:
    f.write("Probability of Damage (PoD) for each Joint:\n")
    for joint, pod in enumerate(PoD, start=1):
        f.write(f"Joint {joint}: PoD = {pod:.2f}\n")
    f.write(f"\nJoints with PoD > {threshold}: {damaged_joints}\n")
print(f"\nPoD results saved at {results_save_path}.")

"""Interpreting PoD:

    High PoD (≈1≈1): Indicates that a large proportion of frames were classified as damaged by the CNN. This is expected for joints where damage is present.
    Low PoD (≈0≈0): Indicates that few frames were classified as damaged. This is expected for joints where no damage is present.

By comparing PoD values across all joints, you can identify which joints are likely damaged based on the model's predictions.

Classification Report:

    Precision: Measures how many predicted positives are true positives. High precision indicates fewer false positives.
    Recall: Measures how many actual positives are identified correctly. High recall indicates fewer false negatives.
    F1-score: The harmonic mean of precision and recall, useful for imbalanced datasets.
    Accuracy: The proportion of correct predictions out of total predictions.

Confusion Matrix:

    Gives a detailed breakdown of True Positives, False Positives, True Negatives, and False Negatives:
        True Positives (TP): Damaged frames classified as damaged.
        False Positives (FP): Undamaged frames classified as damaged.
        True Negatives (TN): Undamaged frames classified as undamaged.
        False Negatives (FN): Damaged frames classified as undamaged.

Saving and Loading Models:

    Persisting Models:
        To avoid retraining models every time, consider saving each trained model.
        Implementation:

Additional Recommendations:

    Model Naming Consistency:
        Ensure that each trained model is saved with a consistent naming convention (e.g., cnn_model_joint1.h5 for Joint 1). This consistency is crucial for correctly loading models during the PoD calculation phase.

    Threshold Selection:
        The threshold for classifying a frame as damaged is set at 0.5. Depending on your model's performance and the desired sensitivity, you might want to adjust this threshold. Analyze the distribution of predicted probabilities to choose an optimal threshold.

    Handling Imbalanced Data:
        If your dataset has an imbalance between undamaged and damaged frames, consider using class weights or data augmentation techniques to balance the training data. This can improve the model's ability to detect damaged frames.

    Error Handling:
        The script includes checks to ensure that training data, models, and test data exist for each joint. You can further enhance error handling by adding try-except blocks to catch and handle unexpected errors during model training or prediction.

    Performance Optimization:
        Training 30 separate models can be time-consuming. If you have access to multiple GPUs or a powerful CPU, consider parallelizing the training process using Python's multiprocessing capabilities or leveraging TensorFlow's distribution strategies.

    Logging:
        For better tracking and debugging, implement logging instead of relying solely on print statements. Python's logging module allows you to record detailed logs with different severity levels (INFO, WARNING, ERROR).
"""

import logging

# Configure logging
logging.basicConfig(filename='training_pod.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Replace print statements with logging
logging.info(f"Processing Joint {joint}...")

"""Saving Training History:

    If you wish to analyze training performance later, consider saving the training history objects.
"""

# Save training history
history_save_path = os.path.join(models_dir, f'cnn_model_joint{joint}_history.npy')
np.save(history_save_path, history.history)
print(f"Training history saved at {history_save_path}.")

"""Model Evaluation Metrics:

    In addition to PoD, consider evaluating other metrics such as Precision, Recall, F1-Score, and the Confusion Matrix to gain deeper insights into your model's performance.
"""

# Classification Report
print(f"\nClassification Report for Joint {joint}:")
print(classification_report(y_test, y_pred, target_names=["Undamaged", "Damaged"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix for Joint {joint}:")
print(cm)

"""Persisting Results:

    For large-scale projects, consider saving PoD results and other metrics to structured formats like CSV or databases for easier querying and analysis.
"""

import pandas as pd

# Create a DataFrame for PoD results
pod_df = pd.DataFrame({
    'Joint': range(1, 31),
    'PoD': PoD
})

# Save to CSV
pod_csv_path = os.path.join(new_test_data_dir, 'PoD_results.csv')
pod_df.to_csv(pod_csv_path, index=False)
print(f"\nPoD results saved as CSV at {pod_csv_path}.")
