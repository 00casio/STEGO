# Patch_dataloader

## 1st Step: Volumes Preprocessing

In this step, we perform the following preprocessing tasks on the volumes:

0. Extract paths for all the patients (img, brain, vessels mask)
1. Train - Validation - Test split
2. Extract spacings and shapes
3. Preprocessing Loop:
    - Load image and mask
    - Crop the image (Optional)
    - Resize segmentation to target spacing
    - Extract slices in which there is brain
    - Standardize Volume 
4. Save the preprocessed volumes for further use.

## 2nd Step: Patch generations

In this step, we extract a collection of 2D patches for each patient:

1. Skull stripping
2. Grid generation (Like in Vessel-CAPTCHA)
3. Simulation of the annotation process
4. Rough mask generation
5. Patches extraction for segmentation 


## 3rd Step (Optional): Visualization

Run dataviz.ipynb once you completed the 2 previous tasks to check and visualized the extracted patches