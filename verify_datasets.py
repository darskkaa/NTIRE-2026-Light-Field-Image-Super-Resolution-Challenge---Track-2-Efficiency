#!/usr/bin/env python3
"""
=============================================================================
NTIRE 2026 LF-SR Challenge - Dataset Verification Script
=============================================================================
Run this script on the VM AFTER downloading datasets from Google Drive.
It verifies that all expected files are present before training.

Usage:
    python verify_datasets.py

Expected dataset structure:
    datasets/
    ├── EPFL/training/              (70 .mat files)
    ├── HCI_new/training/           (20 .mat files)
    ├── HCI_old/training/           (10 .mat files)
    ├── INRIA_Lytro/training/       (35 .mat files)
    └── Stanford_Gantry/training/   (9 .mat files)
                                    ─────────────────
                           Total:   144 .mat files
=============================================================================
"""

import os
import glob
import sys

# ============================================================================
# EXPECTED DATASET MANIFEST
# ============================================================================
EXPECTED_DATASETS = {
    "EPFL": {
        "count": 70,
        "files": [
            "Bench_in_Paris.mat", "Billboards.mat", "Black_Fence.mat", "Bridge.mat",
            "Broken_Mirror.mat", "Bush.mat", "Car_Dashboard.mat", "Caution_Bees.mat",
            "Ceiling_Light.mat", "Chain-link_Fence_1.mat", "Chain-link_Fence_2.mat",
            "Concrete_Cubes.mat", "Danger_de_Mort.mat", "Desktop.mat", "Flowers.mat",
            "Fountain_&_Bench.mat", "Fountain_&_Vincent_1.mat", "Fountain_&_Vincent_2.mat",
            "Fountain_1.mat", "Fountain_2.mat", "Fountain_Pool.mat", "Friends_2.mat",
            "Friends_3.mat", "Friends_4.mat", "Friends_5.mat", "Game_Board.mat",
            "Geometric_Sculpture.mat", "Graffiti.mat", "Gravel_Garden.mat",
            "Houses_&_Lake.mat", "ISO_Chart_12__Decoded.mat", "ISO_Chart_18__Decoded.mat",
            "Mirabelle_Prune_Tree.mat", "Overexposed_Sky.mat", "Parc_du_Luxembourg.mat",
            "Perforated_Metal_1.mat", "Pillars.mat", "Pond_in_Paris.mat",
            "Poppies__Decoded.mat", "Railway_Lines_1.mat", "Railway_Lines_2.mat",
            "Reeds.mat", "Rolex_Learning_Center.mat", "Sewer_Drain.mat",
            "Slab_&_Lake.mat", "Sophie_&_Vincent_1.mat", "Sophie_&_Vincent_2.mat",
            "Sophie_&_Vincent_3.mat", "Sophie_&_Vincent_on_a_Bench.mat",
            "Sophie_&_Vincent_with_Flowers.mat", "Sophie_Krios_&_Vincent.mat",
            "Spear_Fence_1.mat", "Spear_Fence_2.mat", "Stairs.mat",
            "Stone_Pillars_Inside__Decoded.mat", "Stone_Pillars_Outside.mat",
            "Swans_1.mat", "Swans_2.mat", "Tagged_Fence.mat", "Trunk.mat",
            "University.mat", "Vespa__Decoded.mat", "Vine_Wood.mat",
            "Wall_Decoration.mat", "Water_Drops.mat", "Wheat_&_Silos.mat",
            "Wood_&_Net.mat", "Yan_&_Krios_1.mat", "Yan_&_Krios_2.mat",
            "Zwahlen_&_Mayr.mat"
        ]
    },
    "HCI_new": {
        "count": 20,
        "files": [
            "antinous.mat", "boardgames.mat", "boxes.mat", "cotton.mat",
            "dino.mat", "dishes.mat", "greek.mat", "kitchen.mat",
            "medieval2.mat", "museum.mat", "pens.mat", "pillows.mat",
            "platonic.mat", "rosemary.mat", "sideboard.mat", "table.mat",
            "tomb.mat", "tower.mat", "town.mat", "vinyl.mat"
        ]
    },
    "HCI_old": {
        "count": 10,
        "files": [
            "buddha2.mat", "couple.mat", "cube.mat", "horses.mat",
            "maria.mat", "medieval.mat", "papillon.mat", "statue.mat",
            "stillLife.mat", "transparency.mat"
        ]
    },
    "INRIA_Lytro": {
        "count": 35,
        "files": [
            "Bee_2__Decoded.mat", "Bench__Decoded.mat", "BouquetFlower1__Decoded.mat",
            "BouquetFlower2__Decoded.mat", "BouquetFlower3__Decoded.mat",
            "Bridge1__Decoded.mat", "Bridge2__Decoded.mat", "Cactus__Decoded.mat",
            "Chateau1__Decoded.mat", "Chateau2__Decoded.mat", "ChezEdgar__Decoded.mat",
            "Corridor__Decoded.mat", "DistantChurch__Decoded.mat", "Duck__Decoded.mat",
            "Field__Decoded.mat", "Fruits__Decoded.mat", "LeafReflect__Decoded.mat",
            "Leaves__Decoded.mat", "Maison__Decoded.mat", "Mini__Decoded.mat",
            "Panels__Decoded.mat", "Perspective__Decoded.mat", "PlantsIndoor__Decoded.mat",
            "Plushies__Decoded.mat", "Rond__Decoded.mat", "Rose__Decoded.mat",
            "Sign__Decoded.mat", "Statue__Decoded.mat", "Steps__Decoded.mat",
            "Symmetric__Decoded.mat", "TinyMoon__Decoded.mat", "Translucent__Decoded.mat",
            "TreeAndCars__Decoded.mat", "Trees1__Decoded.mat", "Trees2__Decoded.mat"
        ]
    },
    "Stanford_Gantry": {
        "count": 9,
        "files": [
            "Amethyst.mat", "Bracelet.mat", "Chess.mat", "Eucalyptus Flowers.mat",
            "Jelly Beans.mat", "Lego Bulldozer.mat", "Lego Truck.mat",
            "Stanford Bunny.mat", "Treasure Chest.mat"
        ]
    }
}

DATASETS_ROOT = os.path.abspath("datasets")


def verify_datasets():
    """Verify all datasets are present and complete."""
    print("=" * 60)
    print("NTIRE 2026 LF-SR - Dataset Verification")
    print("=" * 60)
    print(f"Checking: {DATASETS_ROOT}")
    print()

    all_passed = True
    total_expected = 0
    total_found = 0

    for ds_name, ds_info in EXPECTED_DATASETS.items():
        training_path = os.path.join(DATASETS_ROOT, ds_name, "training")
        expected_count = ds_info["count"]
        expected_files = set(ds_info["files"])
        total_expected += expected_count

        print(f"[{ds_name}]")

        if not os.path.exists(training_path):
            print(f"  ❌ MISSING: Directory not found: {training_path}")
            all_passed = False
            continue

        # Find all .mat files
        found_files = glob.glob(os.path.join(training_path, "*.mat"))
        found_names = set(os.path.basename(f) for f in found_files)
        found_count = len(found_files)
        total_found += found_count

        if found_count == expected_count:
            print(f"  ✅ PASS: {found_count}/{expected_count} files")
        else:
            print(f"  ⚠️  PARTIAL: {found_count}/{expected_count} files")
            all_passed = False

            # Show missing files
            missing = expected_files - found_names
            if missing:
                print(f"      Missing files:")
                for mf in sorted(missing)[:5]:  # Show first 5
                    print(f"        - {mf}")
                if len(missing) > 5:
                    print(f"        ... and {len(missing) - 5} more")

            # Show extra files (unexpected)
            extra = found_names - expected_files
            if extra:
                print(f"      Extra files (unexpected):")
                for ef in sorted(extra)[:3]:
                    print(f"        + {ef}")

    print()
    print("=" * 60)
    print(f"TOTAL: {total_found}/{total_expected} files")
    print("=" * 60)

    if all_passed:
        print("🎉 ALL DATASETS VERIFIED SUCCESSFULLY!")
        print()
        print("You can now run:")
        print("  ./prepare_data.sh   # Generate training patches")
        print("  ./train.sh          # Start training")
        return 0
    else:
        print("❌ VERIFICATION FAILED - Some datasets are incomplete.")
        print()
        print("Please ensure all files are downloaded and extracted to:")
        print(f"  {DATASETS_ROOT}/<DatasetName>/training/")
        return 1


if __name__ == "__main__":
    sys.exit(verify_datasets())
