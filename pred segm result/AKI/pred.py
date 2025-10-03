import os
import re
import numpy as np
import torch
import tifffile
from transunet_implementation import TransUNet  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
model_path = r"F:\Sample9\TransUnet\Model_Output4\final_model_full.pth"
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()



def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    else:
        return image - mean


def pad_image(image, patch_size):
    h, w = image.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded_image, pad_h, pad_w


def extract_patches(image, patch_size):
    h, w = image.shape
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return patches


def stitch_patches(patches, original_h, original_w, patch_size):
    num_patches_h = (original_h + patch_size - 1) // patch_size
    num_patches_w = (original_w + patch_size - 1) // patch_size
    stitched = np.zeros((num_patches_h * patch_size, num_patches_w * patch_size), dtype=np.uint8)
    for i, patch in enumerate(patches):
        y = (i // num_patches_w) * patch_size
        x = (i % num_patches_w) * patch_size
        stitched[y:y + patch_size, x:x + patch_size] = patch
    return stitched[:original_h, :original_w]



def predict_image(model, image_path, patch_size=256):  
    
    image = tifffile.imread(image_path).astype(np.float32)
    original_h, original_w = image.shape

    
    image_std = standardize_image(image)

   
    padded_image, _, _ = pad_image(image_std, patch_size)
    padded_h, padded_w = padded_image.shape

    
    patches = extract_patches(padded_image, patch_size)
    pred_patches = []

    
    for patch in patches:
        patch_tensor = torch.from_numpy(patch.reshape(1, 1, patch_size, patch_size)).float().to(device)
        with torch.no_grad():
            pred = model(patch_tensor)
            pred_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]  
            pred_patches.append(pred_mask)

    
    stitched_pred = stitch_patches(pred_patches, padded_h, padded_w, patch_size)
    pred_mask = stitched_pred[:original_h, :original_w]

    
    pred_mask = pred_mask + 1
    return pred_mask



if __name__ == "__main__":
   
    image_paths = [
        r"F:\Sample9\Slice\Slice (1).tiff",
        r"F:\Sample9\Slice\Slice (1000).tiff",
        r"F:\Sample9\Slice\Slice (2035).tiff"
    ]

    
    output_dir = r"F:\Sample9\PythonProject\pred\TransUnet\kidney"
    os.makedirs(output_dir, exist_ok=True)

    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        number = re.search(r'\d+', filename).group()
        output_filename = f"pred_{number}.tif"
        output_path = os.path.join(output_dir, output_filename)

       
        pred_mask = predict_image(model, image_path)

        
        tifffile.imwrite(output_path, pred_mask)
   
