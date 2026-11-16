#!/usr/bin/env python3
"""
Download and explore SN-BAS-2025 dataset for external training data.

SN-BAS-2025 contains Team Ball Action Spotting labels:
- Free Kick, Throw In, Goal, Clearance, etc.
- 500+ matches with frame-accurate event timestamps
- This is valid external data (not from the competition game)
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def download_sn_bas_2025(output_dir='external_data/SN-BAS-2025'):
    """Download SN-BAS-2025 dataset."""

    # Check for HF_TOKEN
    token = os.environ.get('HF_TOKEN')
    if not token:
        # Try to load from HF cache (set by `huggingface-cli login`)
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except:
            pass

    if not token:
        print("ERROR: No HuggingFace token found.")
        print("\nTo authenticate:")
        print("1. Get your token from: https://huggingface.co/settings/tokens")
        print("2. Run: export HF_TOKEN=hf_your_token_here")
        print("   OR: huggingface-cli login")
        return False

    print(f"Downloading SN-BAS-2025 to {output_dir}...")
    print("This may take a while (dataset is several GB)...")

    try:
        snapshot_download(
            'SoccerNet/SN-BAS-2025',
            repo_type='dataset',
            local_dir=output_dir,
            token=token,
        )
        print(f"\nDownload complete: {output_dir}")
        return True

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nMake sure you:")
        print("1. Have signed the NDA at: https://huggingface.co/datasets/SoccerNet/SN-BAS-2025")
        print("2. Set HF_TOKEN environment variable with a valid token")
        return False


def explore_dataset(data_dir='external_data/SN-BAS-2025'):
    """Explore SN-BAS-2025 structure after download."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Dataset not found at {data_dir}")
        return

    print(f"\nExploring {data_dir}...")
    print("\nDirectory structure:")
    for item in sorted(data_dir.rglob('*'))[:20]:
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.relative_to(data_dir)} ({size_mb:.1f} MB)")

    # Look for label files
    json_files = list(data_dir.rglob('*.json'))
    print(f"\nFound {len(json_files)} JSON files")

    if json_files:
        # Sample one file
        sample = json_files[0]
        print(f"\nSample file: {sample.relative_to(data_dir)}")
        import json
        with open(sample) as f:
            data = json.load(f)

        print(f"Keys: {list(data.keys())}")
        if 'annotations' in data:
            print(f"Annotations: {len(data['annotations'])} events")
            if data['annotations']:
                print(f"Sample event: {data['annotations'][0]}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true',
                       help='Download SN-BAS-2025')
    parser.add_argument('--explore', action='store_true',
                       help='Explore downloaded dataset')
    parser.add_argument('--output-dir', type=Path,
                       default='external_data/SN-BAS-2025',
                       help='Output directory')
    args = parser.parse_args()

    if args.download:
        success = download_sn_bas_2025(args.output_dir)
        if success and not args.explore:
            explore_dataset(args.output_dir)
    elif args.explore:
        explore_dataset(args.output_dir)
    else:
        parser.print_help()
