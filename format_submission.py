import os
import shutil
import zipfile
from pathlib import Path
import argparse

# NTIRE 2026 Track 2 expects 'Real' and 'Synth' folders.
# Based on standard LF datasets:
# - EPFL, INRIA_Lytro, Stanford_Gantry are Real-world
# - HCI_new, HCI_old are Synthetic

MAPPING = {
    'Real': ['EPFL', 'INRIA_Lytro', 'Stanford_Gantry', 'NTIRE_Val_Real', 'NTIRE_Test_Real'],
    'Synth': ['HCI_new', 'HCI_old', 'NTIRE_Val_Synth', 'NTIRE_Test_Synth']
}

def format_submission(input_dir, output_zip="submission.zip"):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_dir}' not found.")
        return

    # Create temporary structure
    temp_dir = Path("./temp_submission_format")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    real_dir = temp_dir / 'Real'
    synth_dir = temp_dir / 'Synth'
    real_dir.mkdir(parents=True)
    synth_dir.mkdir(parents=True)

    print(f"📦 Formatting submission from: {input_dir}")
    
    real_count = 0
    synth_count = 0

    # Scan the input directory for dataset folders
    for dataset_folder in input_path.iterdir():
        if not dataset_folder.is_dir():
            continue
            
        dataset_name = dataset_folder.name
        
        # Determine if Real or Synth
        target_category = None
        if dataset_name in MAPPING['Real']:
            target_category = 'Real'
        elif dataset_name in MAPPING['Synth']:
            target_category = 'Synth'
        else:
            print(f"⚠️ Warning: Dataset '{dataset_name}' not recognized as Real or Synth. Skipping.")
            continue

        target_base_dir = real_dir if target_category == 'Real' else synth_dir

        # Copy each scene from this dataset into the target category folder
        for scene_folder in dataset_folder.iterdir():
            if not scene_folder.is_dir():
                continue
                
            scene_name = scene_folder.name
            target_scene_dir = target_base_dir / scene_name
            
            # Copy the scene directory
            shutil.copytree(scene_folder, target_scene_dir)
            
            if target_category == 'Real':
                real_count += 1
            else:
                synth_count += 1
                
            print(f"   ✓ Copied {target_category} scene: {scene_name}")

    print(f"\n📊 Summary: {real_count} Real scenes, {synth_count} Synth scenes.")
    
    # Check if meets NTIRE requirements (16 each for Validation/Test set)
    if real_count != 16 or synth_count != 16:
        print(f"\n⚠️ WARNING: NTIRE expects exactly 16 Real and 16 Synth scenes!")
        print(f"   You provided {real_count} Real and {synth_count} Synth.")
        print(f"   (This is expected if you are running inference on the standard 5 datasets instead of the official NTIRE Validation/Test sets).")

    # Zip the contents
    print(f"\n🗜️ Zipping to {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)

    # Cleanup temp dir
    shutil.rmtree(temp_dir)
    print(f"✅ Submission successfully created: {output_zip}")
    print(f"   Run 'python validate_submission.py {output_zip}' to verify.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format inference output for NTIRE 2026 Track 2 submission.")
    parser.add_argument("input_dir", type=str, help="Path to the TEST output directory containing EPFL, HCI_new, etc.")
    parser.add_argument("--output", type=str, default="submission.zip", help="Output ZIP file name")
    args = parser.parse_args()
    
    format_submission(args.input_dir, args.output)
