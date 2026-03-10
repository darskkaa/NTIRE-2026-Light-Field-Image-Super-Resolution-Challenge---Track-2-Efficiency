import zipfile
import sys
import os

def analyze_zip(zip_path):
    print(f"Analyzing {zip_path}")
    if not os.path.exists(zip_path):
        print("File does not exist.")
        return
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            print(f"Total entries: {len(namelist)}")
            
            # Check structure
            has_real = any(f.startswith('Real/') for f in namelist)
            has_synth = any(f.startswith('Synth/') for f in namelist)
            
            print(f"Has Real/ folder: {has_real}")
            print(f"Has Synth/ folder: {has_synth}")
            
            # Check for bad paths
            bad_paths = [f for f in namelist if '\\' in f or f.startswith('/') or f.startswith('./')]
            if bad_paths:
                print(f"WARNING: Found {len(bad_paths)} potentially bad paths!")
                for b in bad_paths[:10]:
                    print(f"  {b}")
            else:
                print("All paths look clean (no backslashes or leading slashes).")
                
            print("\nFirst 10 entries:")
            for e in sorted(namelist)[:10]:
                print(f"  {e}")
                
            print("\nLast 10 entries:")
            for e in sorted(namelist)[-10:]:
                print(f"  {e}")
                
    except Exception as e:
        print(f"Error reading zip: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        zip_path = "submission.zip"
    analyze_zip(zip_path)
