import os
from PIL import Image

# 1. Create a list of all original image paths you want to process.
original_image_paths = [
    r"C:\Users\vidhi\Pictures\Saved Pictures\real img\Student Id Card Design real.jpg"
]

# 2. Define the target directories for both real and fake images.
real_target_dir = os.path.join("cnn_model_dataset", "train", "real")
fake_target_dir = os.path.join("cnn_model_dataset", "train", "fake")

# 3. Create both target directories if they don't already exist.
os.makedirs(real_target_dir, exist_ok=True)
os.makedirs(fake_target_dir, exist_ok=True)

# 4. Loop through each image path in the list.
for image_path in original_image_paths:
    try:
        # 5. Open the original image using Pillow.
        with Image.open(image_path) as img:
            rgb_img = img.convert('RGB')
            original_filename = os.path.basename(image_path)
            
            # 6. Determine the filename and new path.
            if 'real' in original_filename.lower():
                target_dir = real_target_dir
            else:
                # Default to the fake directory if "real" is not in the name
                target_dir = fake_target_dir
            
            # The rest of the logic is the same
            new_filename = os.path.splitext(original_filename)[0] + ".jpg"
            save_path = os.path.join(target_dir, new_filename)
            
            # 7. Save the image in JPG format with high quality.
            rgb_img.save(save_path, 'JPEG', quality=95)
            
            print(f"Image '{original_filename}' successfully saved to '{save_path}'.")
            
    except FileNotFoundError:
        print(f"Error: The file at '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing '{image_path}': {e}")