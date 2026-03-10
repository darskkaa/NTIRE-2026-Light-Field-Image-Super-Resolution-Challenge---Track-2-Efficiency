import zipfile
import os
import struct
import random
import sys

# From validate_submission.py
EXPECTED_REAL_SCENES = 16
EXPECTED_SYNTH_SCENES = 16
EXPECTED_VIEWS_PER_SCENE = 25  # 5x5 angular resolution
EXPECTED_VIEW_NAMES = [f"View_{i}_{j}.bmp" for i in range(5) for j in range(5)]
EXPECTED_REAL_DIMS = (624, 432)   # Width x Height
EXPECTED_SYNTH_DIMS = (500, 500)  # Width x Height
BMP_HEADER_SIZE = 14
BMP_INFO_HEADER_SIZE = 40  # BITMAPINFOHEADER
MIN_PIXEL_MEAN = 20.0    # Images shouldn't be too dark
MAX_PIXEL_MEAN = 235.0   # Images shouldn't be saturated
MIN_PIXEL_STD = 5.0      # Images should have some variance

class ValidationResult:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.stats = {}
        
    def error(self, msg): self.errors.append(f"ERR {msg}")
    def warning(self, msg): self.warnings.append(f"WARN {msg}")
    def info_msg(self, msg): self.info.append(f"INFO {msg}")
    def passed(self): return len(self.errors) == 0

class FileProvider:
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
    def namelist(self): return self._files
    def read(self, filename):
        if self.is_zip: return self.zf.read(filename)
        else:
            with open(os.path.join(self.path, filename), 'rb') as f: return f.read()
    def close(self):
        if self.zf: self.zf.close()

def parse_bmp_header(data):
    if len(data) < 54: return None
    info = {}
    info['magic'] = data[0:2]
    info['file_size'] = struct.unpack('<I', data[2:6])[0]
    info['data_offset'] = struct.unpack('<I', data[10:14])[0]
    info['width'] = struct.unpack('<i', data[18:22])[0]
    info['height'] = struct.unpack('<i', data[22:26])[0]
    info['bits_per_pixel'] = struct.unpack('<H', data[28:30])[0]
    info['compression'] = struct.unpack('<I', data[30:34])[0]
    return info

def analyze_pixel_content(data, bmp_info):
    if bmp_info is None: return None
    offset = bmp_info['data_offset']
    pixel_data = data[offset:]
    if len(pixel_data) == 0: return None
    pixels = list(pixel_data)
    stats = {
        'mean': sum(pixels) / len(pixels),
        'min': min(pixels), 'max': max(pixels), 'std': 0.0,
    }
    mean = stats['mean']
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    stats['std'] = variance ** 0.5
    return stats

def validate_structure(provider, result):
    files = provider.namelist()
    has_real = any(f.startswith('Real/') for f in files)
    has_synth = any(f.startswith('Synth/') for f in files)
    if not has_real: result.error("Missing 'Real/' folder")
    if not has_synth: result.error("Missing 'Synth/' folder")
    return has_real, has_synth

def validate_scenes(provider, result):
    files = provider.namelist()
    real_scenes, synth_scenes = set(), set()
    for f in files:
        parts = f.split('/')
        if len(parts) >= 2:
            folder, scene = parts[0], parts[1]
            if folder == 'Real' and scene: real_scenes.add(scene)
            elif folder == 'Synth' and scene: synth_scenes.add(scene)
    
    if len(real_scenes) != EXPECTED_REAL_SCENES: result.error(f"Expected {EXPECTED_REAL_SCENES} Real scenes, found {len(real_scenes)}")
    if len(synth_scenes) != EXPECTED_SYNTH_SCENES: result.error(f"Expected {EXPECTED_SYNTH_SCENES} Synth scenes, found {len(synth_scenes)}")
    return real_scenes, synth_scenes

def validate_views(provider, result, real_scenes, synth_scenes):
    files = provider.namelist()
    expected_set = set(EXPECTED_VIEW_NAMES)
    all_scenes = [('Real', s) for s in real_scenes] + [('Synth', s) for s in synth_scenes]
    missing_views = []
    
    for folder, scene in all_scenes:
        prefix = f"{folder}/{scene}/"
        scene_files = [f.split('/')[-1] for f in files if f.startswith(prefix) and f.endswith('.bmp')]
        missing = expected_set - set(scene_files)
        if missing: missing_views.append((f"{folder}/{scene}", list(missing)))
    
    if missing_views:
        for scene, views in missing_views[:5]: result.error(f"{scene}/ missing: {views}...")

def validate_bmp_files(provider, result, real_scenes, synth_scenes):
    files = provider.namelist()
    bmp_files = [f for f in files if f.endswith('.bmp') and (f.startswith('Real/') or f.startswith('Synth/'))]
    
    for f in bmp_files:
        try:
            data = provider.read(f)
            bmp_info = parse_bmp_header(data)
            if bmp_info is None or bmp_info['magic'] != b'BM':
                result.error(f"Invalid BMP magic header: {f}")
                continue
            if bmp_info['bits_per_pixel'] != 24: result.error(f"Wrong color depth ({bmp_info['bits_per_pixel']} bpp): {f}")
            if bmp_info['compression'] != 0: result.error(f"Compressed BMP: {f}")
            
            w, h = bmp_info['width'], abs(bmp_info['height'])
            expected_dims = EXPECTED_REAL_DIMS if f.startswith('Real/') else EXPECTED_SYNTH_DIMS
            if (w, h) != expected_dims: result.warning(f"{f}: {w}x{h} (expected {expected_dims})")
        except Exception as e:
            result.error(f"Failed to read {f}: {e}")

def validate_pixel_content(provider, result, sample_size=50):
    files = provider.namelist()
    bmp_files = [f for f in files if f.endswith('.bmp') and (f.startswith('Real/') or f.startswith('Synth/'))]
    sample = random.sample(bmp_files, min(len(bmp_files), sample_size))
    
    for f in sample:
        try:
            data = provider.read(f)
            bmp_info = parse_bmp_header(data)
            if bmp_info is None: continue
            stats = analyze_pixel_content(data, bmp_info)
            if stats is None: continue
            
            if stats['mean'] < MIN_PIXEL_MEAN: result.warning(f"Dark image (mean={stats['mean']:.1f}): {f}")
            if stats['mean'] > MAX_PIXEL_MEAN: result.warning(f"Saturated image (mean={stats['mean']:.1f}): {f}")
            if stats['std'] < MIN_PIXEL_STD: result.warning(f"Low variance (std={stats['std']:.1f}): {f}")
        except Exception:
            pass

def print_summary(result):
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    if result.warnings:
        print(f"\n   WARNINGS ({len(result.warnings)}):")
        for w in result.warnings[:10]: print(f"      {w}")
    if result.errors:
        print(f"\n   ERRORS ({len(result.errors)}):")
        for e in result.errors[:10]: print(f"      {e}")
        print("\nVALIDATION FAILED - DO NOT SUBMIT")
        return False
    else:
        print("\nVALIDATION PASSED - READY TO SUBMIT!")
        return True

def validate_submission_inline(path):
    print("\n" + "="*60)
    print("ULTRA-RIGOROUS SUBMISSION VALIDATOR")
    print("="*60)
    
    result = ValidationResult()
    if not os.path.exists(path):
        print(f"\nERROR: Path not found: {path}")
        return False
    
    print(f"\n   Validating: {path}")
    provider = FileProvider(path)
    
    has_real, has_synth = validate_structure(provider, result)
    if has_real or has_synth:
        real_scenes, synth_scenes = validate_scenes(provider, result)
        validate_views(provider, result, real_scenes, synth_scenes)
        validate_bmp_files(provider, result, real_scenes, synth_scenes)
        validate_pixel_content(provider, result)
        
    provider.close()
    return print_summary(result)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        zip_path = "submission.zip"
    validate_submission_inline(zip_path)
