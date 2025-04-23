import os
import argparse
import subprocess
from pathlib import Path

def convert_psd_to_tiff(source_dir, output_dir):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    for psd_file in source_path.glob("*.psd"):
        output_file = output_path / (psd_file.stem + ".tiff")
        try:
            subprocess.run([
                "magick", str(psd_file),
                str(output_file)
            ], check=True)
            print(f"Converted: {psd_file} -> {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {psd_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert PSD files to TIFF using ImageMagick.")
    parser.add_argument("source_dir", help="Path to the source directory containing PSD files.")
    parser.add_argument("output_dir", help="Path to the output directory for TIFF files.")

    args = parser.parse_args()
    convert_psd_to_tiff(args.source_dir, args.output_dir)

if __name__ == "__main__":
    main()