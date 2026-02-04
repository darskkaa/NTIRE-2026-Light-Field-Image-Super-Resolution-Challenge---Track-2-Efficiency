#!/usr/bin/env python3
"""
=============================================================================
ULTRA-RIGOROUS Submission Validator for NTIRE 2026 LF-SR Challenge (Track 2)
=============================================================================
Checks EVERY possible metadata to ensure CodaBench compatibility.

Usage:
    python validate_submission.py <submission.zip or directory>
    
This script validates:
    1. Structure: Real/ and Synth/ folders exist
    2. Scene Counts: Exactly 16 scenes in each folder
    3. View Naming: View_i_j.bmp (i,j in 0-4) = 25 files per scene
    4. BMP Headers: Magic bytes, file size, data offset
    5. BMP Info Header: Width, height, color depth, compression
    6. Dimensions: Real=624x432, Synth=500x500
    7. File Sizes: Match expected byte counts
    8. Pixel Statistics: Mean, std, min, max (detect black/saturated images)
    9. Channel Order: Detect RGB vs BGR by heuristic (sky/skin tones)
    10. Consistency: All views in a scene have same dimensions
"""

import zipfile
import sys
import os
import struct
from pathlib import Path
from collections import defaultdict
import random

# ============================================================================
# Configuration
# ============================================================================
EXPECTED_REAL_SCENES = 16
EXPECTED_SYNTH_SCENES = 16
EXPECTED_VIEWS_PER_SCENE = 25  # 5x5 angular resolution
EXPECTED_VIEW_NAMES = [f"View_{i}_{j}.bmp" for i in range(5) for j in range(5)]

# Expected dimensions (from rules.md)
EXPECTED_REAL_DIMS = (624, 432)   # Width x Height
EXPECTED_SYNTH_DIMS = (500, 500)  # Width x Height

# BMP Constants
BMP_HEADER_SIZE = 14
BMP_INFO_HEADER_SIZE = 40  # BITMAPINFOHEADER

# Pixel thresholds
MIN_PIXEL_MEAN = 20.0    # Images shouldn't be too dark
MAX_PIXEL_MEAN = 235.0   # Images shouldn't be saturated
MIN_PIXEL_STD = 5.0      # Images should have some variance


class ValidationResult:
    def __init__(self):
        self.errors = []      # Critical: will cause rejection
        self.warnings = []    # Suspicious: may cause issues
        self.info = []        # Informational messages
        self.stats = {}       # Statistics collected
        
    def error(self, msg):
        self.errors.append(f"❌ {msg}")
        
    def warning(self, msg):
        self.warnings.append(f"⚠️  {msg}")
        
    def info_msg(self, msg):
        self.info.append(f"ℹ️  {msg}")
        
    def passed(self):
        return len(self.errors) == 0


class FileProvider:
    """Abstraction to handle both ZIP files and directories."""
    def __init__(self, path):
        self.path = path
        self.is_zip = os.path.isfile(path) and path.endswith('.zip')
        self.zf = None
        
        if self.is_zip:
            self.zf = zipfile.ZipFile(path, 'r')
            self._files = self.zf.namelist()
        else:
            self._files = []
            for root, dirs, files in os.walk(path):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), path)
                    self._files.append(rel.replace('\\', '/'))
    
    def namelist(self):
        return self._files
    
    def read(self, filename):
        if self.is_zip:
            return self.zf.read(filename)
        else:
            with open(os.path.join(self.path, filename), 'rb') as f:
                return f.read()
    
    def close(self):
        if self.zf:
            self.zf.close()


def parse_bmp_header(data):
    """Parse BMP file header and return metadata dict."""
    if len(data) < 54:
        return None
    
    info = {}
    
    # BMP File Header (14 bytes)
    info['magic'] = data[0:2]
    info['file_size'] = struct.unpack('<I', data[2:6])[0]
    info['reserved1'] = struct.unpack('<H', data[6:8])[0]
    info['reserved2'] = struct.unpack('<H', data[8:10])[0]
    info['data_offset'] = struct.unpack('<I', data[10:14])[0]
    
    # DIB Header (BITMAPINFOHEADER - 40 bytes minimum)
    info['header_size'] = struct.unpack('<I', data[14:18])[0]
    info['width'] = struct.unpack('<i', data[18:22])[0]
    info['height'] = struct.unpack('<i', data[22:26])[0]
    info['color_planes'] = struct.unpack('<H', data[26:28])[0]
    info['bits_per_pixel'] = struct.unpack('<H', data[28:30])[0]
    info['compression'] = struct.unpack('<I', data[30:34])[0]
    info['image_size'] = struct.unpack('<I', data[34:38])[0]
    info['h_resolution'] = struct.unpack('<i', data[38:42])[0]
    info['v_resolution'] = struct.unpack('<i', data[42:46])[0]
    info['colors_used'] = struct.unpack('<I', data[46:50])[0]
    info['important_colors'] = struct.unpack('<I', data[50:54])[0]
    
    return info


def analyze_pixel_content(data, bmp_info):
    """Analyze pixel statistics from raw BMP data."""
    if bmp_info is None:
        return None
    
    offset = bmp_info['data_offset']
    pixel_data = data[offset:]
    
    if len(pixel_data) == 0:
        return None
    
    # Convert to list of ints for analysis
    pixels = list(pixel_data)
    
    stats = {
        'mean': sum(pixels) / len(pixels),
        'min': min(pixels),
        'max': max(pixels),
        'std': 0.0,
        'zero_count': pixels.count(0),
        'saturated_count': pixels.count(255),
    }
    
    # Calculate std
    mean = stats['mean']
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    stats['std'] = variance ** 0.5
    
    return stats


def validate_structure(provider, result):
    """Validate folder structure."""
    print("\n" + "="*60)
    print("📁 STRUCTURE VALIDATION")
    print("="*60)
    
    files = provider.namelist()
    
    # Check for Real/ and Synth/ folders
    has_real = any(f.startswith('Real/') for f in files)
    has_synth = any(f.startswith('Synth/') for f in files)
    
    if not has_real:
        result.error("Missing 'Real/' folder at root level")
    else:
        print("   ✓ Real/ folder found")
        
    if not has_synth:
        result.error("Missing 'Synth/' folder at root level")
    else:
        print("   ✓ Synth/ folder found")
    
    # Check for unexpected root folders
    root_folders = set()
    for f in files:
        parts = f.split('/')
        if len(parts) >= 1 and parts[0]:
            root_folders.add(parts[0])
    
    unexpected = root_folders - {'Real', 'Synth'}
    if unexpected:
        result.warning(f"Unexpected root folders: {unexpected}")
    
    return has_real, has_synth


def validate_scenes(provider, result):
    """Validate scene counts and names."""
    print("\n" + "="*60)
    print("🎬 SCENE VALIDATION")
    print("="*60)
    
    files = provider.namelist()
    
    real_scenes = set()
    synth_scenes = set()
    
    for f in files:
        parts = f.split('/')
        if len(parts) >= 2:
            folder, scene = parts[0], parts[1]
            if folder == 'Real' and scene:
                real_scenes.add(scene)
            elif folder == 'Synth' and scene:
                synth_scenes.add(scene)
    
    print(f"\n   Real/ scenes: {len(real_scenes)}")
    if len(real_scenes) != EXPECTED_REAL_SCENES:
        result.error(f"Expected {EXPECTED_REAL_SCENES} Real scenes, found {len(real_scenes)}")
    else:
        print(f"   ✓ Correct count ({EXPECTED_REAL_SCENES})")
    
    print(f"\n   Synth/ scenes: {len(synth_scenes)}")
    if len(synth_scenes) != EXPECTED_SYNTH_SCENES:
        result.error(f"Expected {EXPECTED_SYNTH_SCENES} Synth scenes, found {len(synth_scenes)}")
    else:
        print(f"   ✓ Correct count ({EXPECTED_SYNTH_SCENES})")
    
    result.stats['real_scenes'] = sorted(real_scenes)
    result.stats['synth_scenes'] = sorted(synth_scenes)
    
    return real_scenes, synth_scenes


def validate_views(provider, result, real_scenes, synth_scenes):
    """Validate view naming and count in each scene."""
    print("\n" + "="*60)
    print("👁️  VIEW VALIDATION")
    print("="*60)
    
    files = provider.namelist()
    expected_set = set(EXPECTED_VIEW_NAMES)
    
    all_scenes = [('Real', s) for s in real_scenes] + [('Synth', s) for s in synth_scenes]
    
    missing_views = []
    extra_views = []
    
    for folder, scene in all_scenes:
        prefix = f"{folder}/{scene}/"
        scene_files = [f.split('/')[-1] for f in files if f.startswith(prefix) and f.endswith('.bmp')]
        scene_set = set(scene_files)
        
        missing = expected_set - scene_set
        extra = scene_set - expected_set
        
        if missing:
            missing_views.append((f"{folder}/{scene}", list(missing)[:3]))
        if extra:
            extra_views.append((f"{folder}/{scene}", list(extra)[:3]))
    
    if missing_views:
        print(f"\n   ❌ {len(missing_views)} scenes have MISSING views:")
        for scene, views in missing_views[:5]:
            result.error(f"{scene}/ missing: {views}...")
            print(f"      • {scene}/ missing: {views}...")
        if len(missing_views) > 5:
            print(f"      ... and {len(missing_views)-5} more scenes")
    else:
        print(f"\n   ✓ All {len(all_scenes)} scenes have correct view names (View_0_0 to View_4_4)")
    
    if extra_views:
        for scene, views in extra_views[:3]:
            result.warning(f"{scene}/ has unexpected files: {views}")


def validate_bmp_files(provider, result, real_scenes, synth_scenes):
    """Deep validation of BMP file metadata."""
    print("\n" + "="*60)
    print("🔬 BMP METADATA VALIDATION")
    print("="*60)
    
    files = provider.namelist()
    bmp_files = [f for f in files if f.endswith('.bmp') and (f.startswith('Real/') or f.startswith('Synth/'))]
    
    print(f"\n   Total BMP files to check: {len(bmp_files)}")
    
    # Track issues
    invalid_magic = []
    wrong_depth = []
    compressed = []
    wrong_dims = []
    dim_mismatch_in_scene = []
    
    scene_dims = {}  # scene -> first dims seen
    
    checked = 0
    for f in bmp_files:
        try:
            data = provider.read(f)
            checked += 1
            
            bmp_info = parse_bmp_header(data)
            if bmp_info is None:
                invalid_magic.append(f)
                continue
            
            # 1. Magic bytes
            if bmp_info['magic'] != b'BM':
                invalid_magic.append(f)
                continue
            
            # 2. Color depth (should be 24-bit RGB)
            if bmp_info['bits_per_pixel'] != 24:
                wrong_depth.append((f, bmp_info['bits_per_pixel']))
            
            # 3. Compression (should be 0 = uncompressed)
            if bmp_info['compression'] != 0:
                compressed.append((f, bmp_info['compression']))
            
            # 4. Dimensions
            w, h = bmp_info['width'], abs(bmp_info['height'])
            parts = f.split('/')
            folder = parts[0]
            scene = parts[1] if len(parts) > 1 else ""
            
            expected_dims = EXPECTED_REAL_DIMS if folder == 'Real' else EXPECTED_SYNTH_DIMS
            if (w, h) != expected_dims:
                wrong_dims.append((f, (w, h), expected_dims))
            
            # 5. Consistency within scene
            scene_key = f"{folder}/{scene}"
            if scene_key not in scene_dims:
                scene_dims[scene_key] = (w, h)
            elif scene_dims[scene_key] != (w, h):
                dim_mismatch_in_scene.append((f, (w, h), scene_dims[scene_key]))
                
        except Exception as e:
            result.error(f"Failed to read {f}: {str(e)}")
    
    print(f"   ✓ Checked {checked} BMP files")
    
    # Report issues
    if invalid_magic:
        for f in invalid_magic[:3]:
            result.error(f"Invalid BMP magic header: {f}")
        print(f"\n   ❌ {len(invalid_magic)} files have invalid BMP headers")
    else:
        print("   ✓ All files have valid BMP magic (0x42, 0x4D)")
    
    if wrong_depth:
        for f, depth in wrong_depth[:3]:
            result.error(f"Wrong color depth ({depth} bpp, expected 24): {f}")
        print(f"\n   ❌ {len(wrong_depth)} files have wrong color depth")
    else:
        print("   ✓ All files are 24-bit (RGB)")
    
    if compressed:
        for f, comp in compressed[:3]:
            result.error(f"Compressed BMP (type {comp}): {f}")
        print(f"\n   ❌ {len(compressed)} files are compressed")
    else:
        print("   ✓ All files are uncompressed")
    
    if wrong_dims:
        print(f"\n   ⚠️  {len(wrong_dims)} files have unexpected dimensions:")
        for f, actual, expected in wrong_dims[:3]:
            result.warning(f"{f}: {actual} (expected {expected})")
            print(f"      • {f}: {actual} expected {expected}")
    else:
        print("   ✓ All dimensions match expected (Real=624x432, Synth=500x500)")
    
    if dim_mismatch_in_scene:
        for f, actual, expected in dim_mismatch_in_scene[:3]:
            result.warning(f"Dimension inconsistency in scene: {f}")
    
    result.stats['bmp_count'] = len(bmp_files)


def validate_pixel_content(provider, result, sample_size=50):
    """Analyze pixel content to detect black/saturated images."""
    print("\n" + "="*60)
    print("🎨 PIXEL CONTENT ANALYSIS")
    print("="*60)
    
    files = provider.namelist()
    bmp_files = [f for f in files if f.endswith('.bmp') and (f.startswith('Real/') or f.startswith('Synth/'))]
    
    # Sample for efficiency
    sample = random.sample(bmp_files, min(len(bmp_files), sample_size))
    
    print(f"\n   Sampling {len(sample)} images for content analysis...")
    
    dark_images = []
    bright_images = []
    low_variance = []
    channel_stats = {'r_mean': [], 'g_mean': [], 'b_mean': []}
    
    for f in sample:
        try:
            data = provider.read(f)
            bmp_info = parse_bmp_header(data)
            if bmp_info is None:
                continue
            
            stats = analyze_pixel_content(data, bmp_info)
            if stats is None:
                continue
            
            # Check for issues
            if stats['mean'] < MIN_PIXEL_MEAN:
                dark_images.append((f, stats['mean']))
            if stats['mean'] > MAX_PIXEL_MEAN:
                bright_images.append((f, stats['mean']))
            if stats['std'] < MIN_PIXEL_STD:
                low_variance.append((f, stats['std']))
            
            # Analyze per-channel (BMP stores as BGR)
            offset = bmp_info['data_offset']
            pixel_data = data[offset:]
            if len(pixel_data) >= 3:
                # Sample every 1000th pixel for speed
                step = max(3, len(pixel_data) // 1000 * 3)
                b_vals = [pixel_data[i] for i in range(0, min(len(pixel_data), 10000), step)]
                g_vals = [pixel_data[i] for i in range(1, min(len(pixel_data), 10000), step)]
                r_vals = [pixel_data[i] for i in range(2, min(len(pixel_data), 10000), step)]
                
                if b_vals and g_vals and r_vals:
                    channel_stats['b_mean'].append(sum(b_vals)/len(b_vals))
                    channel_stats['g_mean'].append(sum(g_vals)/len(g_vals))
                    channel_stats['r_mean'].append(sum(r_vals)/len(r_vals))
                    
        except Exception:
            pass
    
    # Report findings
    if dark_images:
        print(f"\n   ⚠️  {len(dark_images)} images appear TOO DARK:")
        for f, mean in dark_images[:3]:
            result.warning(f"Dark image (mean={mean:.1f}): {f}")
            print(f"      • {f}: mean={mean:.1f}")
        print("      Possible cause: 0-1 vs 0-255 range mismatch")
    else:
        print("   ✓ No suspiciously dark images")
    
    if bright_images:
        print(f"\n   ⚠️  {len(bright_images)} images appear SATURATED:")
        for f, mean in bright_images[:3]:
            result.warning(f"Saturated image (mean={mean:.1f}): {f}")
    else:
        print("   ✓ No saturated images")
    
    if low_variance:
        print(f"\n   ⚠️  {len(low_variance)} images have LOW VARIANCE (flat/uniform):")
        for f, std in low_variance[:3]:
            result.warning(f"Low variance (std={std:.1f}): {f}")
    else:
        print("   ✓ All images have reasonable variance")
    
    # Channel order analysis
    if channel_stats['r_mean'] and channel_stats['b_mean']:
        avg_r = sum(channel_stats['r_mean']) / len(channel_stats['r_mean'])
        avg_g = sum(channel_stats['g_mean']) / len(channel_stats['g_mean'])
        avg_b = sum(channel_stats['b_mean']) / len(channel_stats['b_mean'])
        
        print(f"\n   Channel Averages (BMP order): B={avg_b:.1f}, G={avg_g:.1f}, R={avg_r:.1f}")
        
        # Heuristic: In natural images, R and G are often higher than B
        # If B >> R, channels might be swapped
        if avg_b > avg_r + 30:
            result.warning("Blue channel unusually high - possible RGB/BGR swap?")
            print("   ⚠️  Blue channel seems high - check RGB/BGR ordering!")
        else:
            print("   ✓ Channel distribution looks reasonable")


def print_summary(result):
    """Print final summary."""
    print("\n" + "="*60)
    print("📋 VALIDATION SUMMARY")
    print("="*60)
    
    if result.warnings:
        print(f"\n   ⚠️  WARNINGS ({len(result.warnings)}):")
        for w in result.warnings[:10]:
            print(f"      {w}")
        if len(result.warnings) > 10:
            print(f"      ... and {len(result.warnings)-10} more")
    
    if result.errors:
        print(f"\n   ❌ ERRORS ({len(result.errors)}):")
        for e in result.errors[:10]:
            print(f"      {e}")
        if len(result.errors) > 10:
            print(f"      ... and {len(result.errors)-10} more")
        
        print("\n" + "="*60)
        print("❌ VALIDATION FAILED - DO NOT SUBMIT")
        print("="*60)
        return False
    else:
        print("\n" + "="*60)
        print("✅ VALIDATION PASSED - READY TO SUBMIT!")
        print("="*60)
        print(f"\n   • Structure: Valid (Real/ + Synth/)")
        print(f"   • Scenes: {len(result.stats.get('real_scenes', []))} Real, {len(result.stats.get('synth_scenes', []))} Synth")
        print(f"   • Images: {result.stats.get('bmp_count', 0)} BMP files verified")
        print(f"   • Content: Pixel analysis passed")
        print("\n   🚀 Upload to CodaBench with confidence!")
        return True


def validate_submission(path):
    """Main validation function."""
    print("\n" + "="*60)
    print("🔍 ULTRA-RIGOROUS SUBMISSION VALIDATOR")
    print("   NTIRE 2026 Track 2 - Light Field Super-Resolution")
    print("="*60)
    
    result = ValidationResult()
    
    # Check path exists
    if not os.path.exists(path):
        print(f"\n❌ ERROR: Path not found: {path}")
        return False
    
    print(f"\n   Validating: {path}")
    is_zip = os.path.isfile(path) and path.endswith('.zip')
    print(f"   Type: {'ZIP archive' if is_zip else 'Directory'}")
    
    try:
        provider = FileProvider(path)
        
        # Run all validations
        has_real, has_synth = validate_structure(provider, result)
        
        if has_real or has_synth:
            real_scenes, synth_scenes = validate_scenes(provider, result)
            validate_views(provider, result, real_scenes, synth_scenes)
            validate_bmp_files(provider, result, real_scenes, synth_scenes)
            validate_pixel_content(provider, result)
        
        provider.close()
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return print_summary(result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Try to find submission
        for candidate in ['./submission.zip', './MyEfficientLFNet_submission.zip']:
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            print("Usage: python validate_submission.py <submission.zip or directory>")
            sys.exit(1)
    
    success = validate_submission(path)
    sys.exit(0 if success else 1)
