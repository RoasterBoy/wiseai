import os
import argparse
import subprocess

def convert_psd_to_jpg(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".psd"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_dir, output_filename)

            try:
                subprocess.run([
                    "magick", "convert",  # Use "convert" if older ImageMagick version
                    input_path,
                    "-quality", "95",  # High quality setting
                    output_path
                ], check=True)
                print(f"Converted: {filename} → {output_filename}")
            except subprocess.CalledProcessError:
                print(f"Failed to convert: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Convert PSD files to high-quality JPGs.")
    parser.add_argument("input_dir", help="Directory containing PSD files")
    parser.add_argument("output_dir", help="Directory to save JPG files")
    args = parser.parse_args()

    convert_psd_to_jpg(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()