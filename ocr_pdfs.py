#!/usr/bin/env python3
import os
import argparse
import subprocess

def process_pdf(input_file, output_file, lang="eng", tesseract_options=""):
    """
    Runs ocrmypdf on a single PDF file with additional Tesseract options.
    """
    # Build the command with Tesseract OCR options if provided.
    cmd = ["ocrmypdf"]
    if tesseract_options:
        cmd += ["--tesseract-ocr-options", tesseract_options]
    cmd += ["-l", lang, input_file, output_file]
    
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description="OCR all PDF files in a directory and save the output to another directory."
    )
    parser.add_argument("input_dir", help="Path to the directory containing PDF files to OCR")
    parser.add_argument("output_dir", help="Path to the directory to save OCR’d PDF files")
    parser.add_argument("--lang", default="eng", help="OCR language (default: eng)")
    parser.add_argument("--psm", help="Tesseract Page Segmentation Mode (e.g., '6' for a uniform block of text)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # If a PSM value is provided, prepare the Tesseract options string.
    tesseract_options = f"--psm {args.psm}" if args.psm else ""

    # Process each PDF file in the input directory.
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(".pdf"):
            input_file = os.path.join(args.input_dir, filename)
            output_file = os.path.join(args.output_dir, filename)
            print(f"Processing: {input_file}")
            try:
                process_pdf(input_file, output_file, args.lang, tesseract_options)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    main()