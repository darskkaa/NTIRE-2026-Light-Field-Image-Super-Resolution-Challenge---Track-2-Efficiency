
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import os

def check_channels():
    # Find a sample image
    search_path = "./log/SR_5x5_4x/NTIRE_Val_Real/MyEfficientLFNet/results/TEST/NTIRE_Val_Real/"
    sample_file = None
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file.endswith(".bmp"):
                sample_file = os.path.join(root, file)
                break
        if sample_file: break
    
    if not sample_file:
        print("No output images found to check.")
        return

    print(f"Checking: {sample_file}")
    img = imageio.imread(sample_file)
    
    # Create Swapped version (RGB <-> BGR)
    img_swapped = img[:, :, ::-1]
    
    # Save both for user to inspect (or we can compute stats)
    imageio.imwrite("check_original.png", img)
    imageio.imwrite("check_swapped.png", img_swapped)
    
    print("Saved 'check_original.png' and 'check_swapped.png'.")
    print("Please download/view them. The one with CORRECT colors is the right format.")
    print("If 'check_original' looks blue-ish/wrong, we need to flip channels.")

if __name__ == "__main__":
    check_channels()
