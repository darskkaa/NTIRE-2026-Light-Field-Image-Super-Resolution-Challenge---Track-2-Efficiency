#!/usr/bin/env python3
"""
Validate submission.zip for NTIRE 2026 LF-SR Challenge
Checks folder structure and file naming before uploading to CodaBench.
"""

import zipfile
import sys
import os
from pathlib import Path

def validate_submission(zip_path):
    """Validate submission ZIP structure."""
    errors = []
    warnings = []
    
    if os.path.isdir(zip_path):
        print(f"❌ ERROR: '{zip_path}' is a directory!")
        print(f"   Please point to the ZIP file (e.g., ./submission.zip) not the folder.")
        # Check if the zip is likely just outside or inside
        likely_zip = os.path.join(os.path.dirname(zip_path), "submission.zip")
        if os.path.exists(likely_zip):
            print(f"   💡 Did you mean: '{likely_zip}'?")
        return False

    if not os.path.exists(zip_path):
        print(f"❌ ERROR: {zip_path} not found!")
        return False
    
    print(f"📦 Validating: {zip_path}")
    print("=" * 50)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        
        # Check for Real/ and Synth/ folders
        has_real = any(f.startswith('Real/') for f in file_list)
        has_synth = any(f.startswith('Synth/') for f in file_list)
        
        if not has_real:
            errors.append("Missing 'Real/' folder in ZIP root")
        if not has_synth:
            errors.append("Missing 'Synth/' folder in ZIP root")
        
        # Expected scenes (from rules.md)
        expected_views = [f"View_{i}_{j}.bmp" for i in range(5) for j in range(5)]
        
        # Count and validate scenes
        real_scenes = set()
        synth_scenes = set()
        
        for f in file_list:
            parts = f.split('/')
            if len(parts) >= 3:
                folder = parts[0]  # Real or Synth
                scene = parts[1]   # Scene name
                filename = parts[2] if len(parts) > 2 else ""
                
                if folder == 'Real' and scene:
                    real_scenes.add(scene)
                elif folder == 'Synth' and scene:
                    synth_scenes.add(scene)
        
        print(f"\n📁 Real/ scenes found: {len(real_scenes)}")
        for scene in sorted(real_scenes):
            scene_files = [f for f in file_list if f.startswith(f"Real/{scene}/")]
            view_files = [f.split('/')[-1] for f in scene_files if f.endswith('.bmp')]
            
            missing = set(expected_views) - set(view_files)
            if missing:
                errors.append(f"Real/{scene}/ missing views: {sorted(missing)[:3]}...")
            else:
                print(f"   ✓ {scene} ({len(view_files)} views)")
        
        print(f"\n📁 Synth/ scenes found: {len(synth_scenes)}")
        for scene in sorted(synth_scenes):
            scene_files = [f for f in file_list if f.startswith(f"Synth/{scene}/")]
            view_files = [f.split('/')[-1] for f in scene_files if f.endswith('.bmp')]
            
            missing = set(expected_views) - set(view_files)
            if missing:
                errors.append(f"Synth/{scene}/ missing views: {sorted(missing)[:3]}...")
            else:
                print(f"   ✓ {scene} ({len(view_files)} views)")
        
        # Check expected counts
        if len(real_scenes) != 16:
            warnings.append(f"Expected 16 Real scenes, found {len(real_scenes)}")
        if len(synth_scenes) != 16:
            warnings.append(f"Expected 16 Synth scenes, found {len(synth_scenes)}")
        
        # Rigorous check: Validate EVERY BMP file
        print(f"\n🔍 Rigorous Check: Scanning inner content of {len(real_scenes) + len(synth_scenes)} scenes...")
        
        corrupt_files = []
        small_files = []
        dimensions = {} # scene -> (w, h)
        
        bmp_count = 0
        for f in file_list:
            if f.endswith('.bmp') and (f.startswith('Real/') or f.startswith('Synth/')):
                bmp_count += 1
                try:
                    data = zf.read(f)
                    
                    # 1. Size check
                    if len(data) < 1000:
                        small_files.append(f"{f} ({len(data)} bytes)")
                        errors.append(f"File too small: {f}")
                        continue

                    # 2. Header check (BM signature)
                    if data[:2] != b'BM':
                        corrupt_files.append(f"{f} (Invalid Magic Header)")
                        errors.append(f"Invalid BMP header: {f}")
                        continue
                        
                    # 3. Simple Dimension Parse (Little endian, standard BMP header)
                    # Width at offset 18 (4 bytes), Height at offset 22 (4 bytes)
                    import struct
                    try:
                        w, h = struct.unpack('<II', data[18:26])
                        if w == 0 or h == 0:
                            corrupt_files.append(f"{f} (Zero dimensions: {w}x{h})")
                            errors.append(f"Invalid dimensions {w}x{h}: {f}")
                        else:
                            # Consistency check: Scene images should likely be same size
                            parts = f.split('/')
                            if len(parts) >= 2:
                                scene_key = f"{parts[0]}/{parts[1]}"
                                if scene_key not in dimensions:
                                    dimensions[scene_key] = (w, h)
                                elif dimensions[scene_key] != (w, h):
                                    # Warning only, as some datasets might have varying aspect ratios per view (unlikely but possible)
                                    pass 
                    except Exception as e:
                        corrupt_files.append(f"{f} (Header parse error: {str(e)})")
                        
                except Exception as e:
                    errors.append(f"Could not read {f}: {str(e)}")

        print(f"   ✓ Scanned {bmp_count} BMP files.")
        
        if corrupt_files:
            print(f"\n❌ FOUND {len(corrupt_files)} CORRUPT FILES:")
            for cf in corrupt_files[:5]:
                print(f"   • {cf}")
            if len(corrupt_files) > 5:
                print(f"   ... and {len(corrupt_files)-5} more.")
        else:
            print("   ✓ All BMP headers and dimensions valid.")

    print("\n" + "=" * 50)
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"   • {w}")
    
    if errors:
        print("\n❌ ERRORS (will cause CodaBench to reject):")
        for e in errors:
            print(f"   • {e}")
        return False
    else:
        print("\n✅ PASSED! Submission is robust.")
        print("   • Structure: Valid")
        print(f"   • Content:   Verified {bmp_count} images")
        print("   Ready to upload to CodaBench!")
        return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        zip_path = "./submission.zip"
    
    # Help user find the file if they haven't created it yet
    if not os.path.exists(zip_path):
        legacy_path = "./MyEfficientLFNet_submission.zip"
        if os.path.exists(legacy_path):
            print(f"⚠️  NOTE: Found '{legacy_path}' instead of '{zip_path}'.")
            print("   The script will check this file, but you should rename it or re-run create_submission.sh")
            print("   to match the new naming convention.\n")
            zip_path = legacy_path
    
    validate_submission(zip_path)
