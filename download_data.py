
import subprocess
import sys
import os
import shutil

def install_gdown():
    print("Installing gdown...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

def download_and_extract():
    import gdown
    
    url = 'https://drive.google.com/drive/folders/1LfPTTTtTDOPyNg3D-B_RfzwBZd4D0-HH?usp=drive_link'
    output_folder = 'data_for_inference'
    
    if os.path.exists(output_folder):
        print(f"Directory {output_folder} already exists. Skipping download to avoid duplicates. Please remove it if you want to re-download.")
        return

    print(f"Downloading folder to {output_folder}...")
    # gdown.download_folder(url, output=output_folder, quiet=False, use_cookies=False)
    # Using CLI due to some issues with python api sometimes with folders
    subprocess.check_call([sys.executable, "-m", "gdown", "--folder", url, "-O", output_folder])

    print("Download complete.")

if __name__ == "__main__":
    try:
        import gdown
    except ImportError:
        install_gdown()
    
    download_and_extract()
