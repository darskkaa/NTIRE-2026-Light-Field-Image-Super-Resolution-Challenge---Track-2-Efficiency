"""
Quick diagnostic: dump submission.zip contents and check one BMP file.
Run this on the VM AFTER generating submission.zip to verify the format.
"""
import zipfile
import struct
import sys
import os

zip_path = sys.argv[1] if len(sys.argv) > 1 else "submission.zip"

if not os.path.exists(zip_path):
    print(f"❌ {zip_path} not found!")
    sys.exit(1)

print(f"=== Diagnosing {zip_path} ({os.path.getsize(zip_path)} bytes) ===\n")

with zipfile.ZipFile(zip_path, 'r') as zf:
    entries = zf.namelist()
    
    print(f"Total entries in zip: {len(entries)}")
    
    # Check for backslashes
    has_backslash = any('\\' in e for e in entries)
    print(f"Contains backslashes: {has_backslash}")
    if has_backslash:
        print("  ❌ CRITICAL: Backslashes found! CodaBench expects forward slashes!")
        for e in entries[:5]:
            if '\\' in e:
                print(f"    Example: {e}")
    
    # Check structure
    top_level = set()
    real_scenes = set()
    synth_scenes = set()
    bmp_count = 0
    
    for e in entries:
        parts = e.split('/')
        if parts[0]:
            top_level.add(parts[0])
        if len(parts) >= 3 and parts[-1].endswith('.bmp'):
            bmp_count += 1
            if parts[0] == 'Real':
                real_scenes.add(parts[1])
            elif parts[0] == 'Synth':
                synth_scenes.add(parts[1])
    
    print(f"\nTop-level items in zip: {sorted(top_level)}")
    print(f"  Expected: ['Real', 'Synth']")
    
    has_real = 'Real' in top_level
    has_synth = 'Synth' in top_level
    print(f"\nHas Real/: {has_real}")
    print(f"Has Synth/: {has_synth}")
    
    print(f"\nReal scenes ({len(real_scenes)}): {sorted(real_scenes)}")
    print(f"Synth scenes ({len(synth_scenes)}): {sorted(synth_scenes)}")
    print(f"Total BMP files: {bmp_count}")
    
    # Print ALL entries
    print(f"\n=== ALL {len(entries)} ZIP ENTRIES ===")
    for e in sorted(entries):
        info = zf.getinfo(e)
        print(f"  {e}  ({info.file_size} bytes)")
    
    # Check one BMP file
    bmp_entries = [e for e in entries if e.endswith('.bmp')]
    if bmp_entries:
        first_bmp = bmp_entries[0]
        data = zf.read(first_bmp)
        print(f"\n=== BMP CHECK: {first_bmp} ===")
        print(f"  File size: {len(data)} bytes")
        if len(data) >= 54:
            magic = data[0:2]
            width = struct.unpack('<i', data[18:22])[0]
            height = struct.unpack('<i', data[22:26])[0]
            bpp = struct.unpack('<H', data[28:30])[0]
            compression = struct.unpack('<I', data[30:34])[0]
            print(f"  Magic: {magic}")
            print(f"  Width: {width}, Height: {abs(height)}")
            print(f"  Bits per pixel: {bpp}")
            print(f"  Compression: {compression}")
            
            if magic != b'BM':
                print("  ❌ Invalid BMP header!")
            if bpp != 24:
                print(f"  ❌ Expected 24-bit, got {bpp}-bit!")
            if compression != 0:
                print(f"  ❌ Compressed BMP! Expected uncompressed (0)")
        else:
            print("  ❌ File too small to be a valid BMP!")
    
    # Final verdict
    print("\n=== VERDICT ===")
    if has_real and has_synth and bmp_count > 0 and not has_backslash:
        print("✅ Format looks correct")
    else:
        print("❌ Format issues detected!")
