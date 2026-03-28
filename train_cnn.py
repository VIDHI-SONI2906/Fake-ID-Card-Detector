import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# ====================================================================
#  1. Define File Paths and Directories
# ====================================================================

# Base directory for the CNN dataset
dataset_base_dir = 'cnn_model_dataset'

# Directories for training data
train_dir = os.path.join(dataset_base_dir, 'train')

# Directories for validation data
val_dir = os.path.join(dataset_base_dir, 'val')
val_real_dir = os.path.join(val_dir, 'real')
val_fake_dir = os.path.join(val_dir, 'fake')

# ====================================================================
#  2. Data Preparation: Copy Training Images to Validation Folders
# ====================================================================

def setup_validation_data():
    """Copies images from the train directory to the validation directory."""
    print("Setting up validation data...")
    
    # Ensure validation directories exist
    os.makedirs(val_real_dir, exist_ok=True)
    os.makedirs(val_fake_dir, exist_ok=True)
    
    # Copy images from train/real to val/real
    source_real = os.path.join(train_dir, 'real')
    dest_real = val_real_dir
    for filename in os.listdir(source_real):
        shutil.copy(os.path.join(source_real, filename), os.path.join(dest_real, filename))
        
    # Copy images from train/fake to val/fake
    source_fake = os.path.join(train_dir, 'fake')
    dest_fake = val_fake_dir
    for filename in os.listdir(source_fake):
        shutil.copy(os.path.join(source_fake, filename), os.path.join(dest_fake, filename))
    
    print("Validation data successfully set up.")

# ====================================================================
#  3. Main Execution: Set up Data and Train the Model
# ====================================================================

if __name__ == '__main__':
    # Step A: Populate the validation folders with images from the training folders
    setup_validation_data()

    print("\n--- Starting CNN Model Training ---")

    # Step B: Define image data generators to load images from the folders
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load images from the training directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=1, # Use batch_size=1 since you have one image in each folder
        class_mode='binary'
    )
    
    # Load images from the validation directory
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=1, # Use batch_size=1 since you have one image in each folder
        class_mode='binary'
    )

    # Step C: Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Step D: Train the model
    history = model.fit(
        train_generator,
        epochs=5, # Train for 5 epochs
        validation_data=validation_generator
    )

    # Step E: Save the trained model
    model_save_path = 'model.h5'
    model.save(model_save_path)
    print(f"\nModel saved successfully as '{model_save_path}'")