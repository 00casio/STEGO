import os
from PIL import Image
import numpy as np
import scipy.io as sio

def convert_image_to_mat(image_path, output_path):
    # Load the image
    img = Image.open(image_path).convert('L')
    
    # Convert the image to numpy array
    img_array = np.array(img)
    
    # Map the values 0 to 0 and 255 to 3    
    img_array = np.where(img_array == 0, 1, img_array)

    img_array = np.where(img_array == 255, 0, img_array)
    # Ensure the type is int32
    img_array = img_array.astype(np.int32)
    
    # Prepare the dictionary for saving as .mat file
    mat_data = {'imageData': img_array}
    
    # Save as .mat file
    sio.savemat(output_path, mat_data)

def batch_process_images(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mat')
            convert_image_to_mat(input_path, output_path)



# # the input folder is the folder containing the .jpj images that we will fix their pixels values, changing 255
# # to 1 and keeping 0 as it is, this is the case for two classes only, 0 for the first class and 1 for the second class
# # the function will save the fixed images in the gt file directly. 
input_folder = 'your_labels_path_before_fixing_pixels'
output_folder = '../../dataset4/potsdam/gt' # the output folder is the folder where the fixed images will be saved in the gt folder
batch_process_images(input_folder, output_folder)
