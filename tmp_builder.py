import re

with open(r"C:\Users\darkz\.gemini\antigravity\brain\d3b67f75-b369-4e08-8c14-e3783f14330a\standalone_colab_submission.py", "r") as f:
    standalone = f.read()

with open(r"C:\Users\darkz\.gemini\antigravity\scratch\BasicLFSR\validate_submission.py", "r", encoding='utf-8') as f:
    validate = f.read()

new_pip = """import sys
import os
import subprocess
import shutil
import zipfile
import struct
import random
from pathlib import Path
from collections import defaultdict

# Setup pip installs
print("Installing requirements...")
try:
    import einops
    import fvcore
    import h5py
    import imageio
    from skimage import metrics
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu121"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "causal-conv1d==1.4.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mamba-ssm==2.2.2"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fvcore"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers<4.40.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown", "h5py", "scipy", "imageio", "scikit-image", "einops"])

"""
idx = standalone.find("import gdown")
standalone = new_pip + standalone[idx:]

validate_func = validate[validate.find("EXPECTED_REAL_SCENES = 16"):validate.find("if __name__ == \"__main__\":")]

standalone = standalone.replace("format_submission('./Results', 'ntire_submission.zip')", 
    "format_submission('./Results', 'ntire_submission.zip')\n    print('\\n[5/5] Validating Submission Zip...')\n    validate_submission('ntire_submission.zip')")

output_code = standalone + "\n\n" + validate_func

with open(r"C:\Users\darkz\.gemini\antigravity\scratch\BasicLFSR\run_ntire_colab_submission.py", "w", encoding='utf-8') as f:
    f.write(output_code)

print("Created run_ntire_colab_submission.py successfully!")
