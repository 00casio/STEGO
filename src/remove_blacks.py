import os
import numpy as np
from PIL import Image

# Input and output directories
input_dir = 'path_to_images_with_black_backgrounds'
output_dir = '../../dataset/potsdam/imgs'


# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        # Load grayscale image
        gray_image = Image.open(os.path.join(input_dir, filename)).convert('L')
        
        # Convert to numpy array
        gray_array = np.array(gray_image)
        
        # Generate random values for pixels with value 0
        zero_indices = np.where(gray_array == 0)
        modified_array = gray_array.copy()
        for i, j in zip(*zero_indices):
            modified_array[i, j] = np.random.randint(30, 61)
        
        # Convert modified array back to PIL Image
        modified_image = Image.fromarray(modified_array.astype('uint8'), 'L')
        
        # Convert to RGB
        rgb_image = modified_image.convert('RGB')
        
        # Save the modified image to the output directory
        output_path = os.path.join(output_dir, filename)
        rgb_image.save(output_path)
        
        print(f'{filename} processed and saved to {output_path}')

print('All images processed.')
