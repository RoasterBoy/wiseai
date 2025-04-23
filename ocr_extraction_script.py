import os
import json
import argparse
import re
import pytesseract
from PIL import Image
from psd_tools import PSDImage
from pathlib import Path

def extract_text_with_ocr(image, name_pattern, date_pattern, location_pattern):
    """Extract text from image using OCR and categorize it"""
    # Run OCR on the image
    try:
        extracted_text = pytesseract.image_to_string(image)
        
        # Split by newlines to get separate text elements
        text_lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        # Initialize metadata
        metadata = {
            'name': '',
            'dates': [],
            'location': '',
            'notes': []
        }
        
        # Categorize each line of text
        for line in text_lines:
            if re.match(name_pattern, line):
                metadata['name'] = line
            elif re.search(date_pattern, line):
                dates = re.findall(date_pattern, line)
                metadata['dates'].extend(dates)
            elif re.search(location_pattern, line):
                metadata['location'] = line
            else:
                # If this line doesn't match any pattern, add it to notes
                metadata['notes'].append(line)
        
        return True, text_lines, metadata
    except Exception as e:
        return False, [], {'error': str(e)}

def process_psd_file(psd_path, output_dir, name_pattern, date_pattern, location_pattern):
    """Process a single PSD file and extract its content using OCR"""
    try:
        # Open the PSD file
        psd = PSDImage.open(psd_path)
        
        # Create a base filename without extension
        base_filename = os.path.splitext(os.path.basename(psd_path))[0]
        
        # Extract the composite image
        image = psd.composite()
        image_path = os.path.join(output_dir, 'images', f"{base_filename}.png")
        image.save(image_path)
        
        # Extract text using OCR
        success, text_lines, metadata = extract_text_with_ocr(
            image, name_pattern, date_pattern, location_pattern
        )
        
        if not success:
            return False, f"OCR failed for {psd_path}: {metadata.get('error', 'Unknown error')}"
        
        # Add file path to metadata
        metadata['original_file'] = str(psd_path)
        
        # Save all extracted text to a file
        text_path = os.path.join(output_dir, 'text', f"{base_filename}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))
        
        # Save structured metadata as JSON
        metadata_path = os.path.join(output_dir, 'metadata', f"{base_filename}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True, base_filename
            
    except Exception as e:
        return False, f"Error processing {psd_path}: {str(e)}"

def process_image_file(image_path, output_dir, name_pattern, date_pattern, location_pattern):
    """Process a single image file and extract its content using OCR"""
    try:
        # Open the image file
        image = Image.open(image_path)
        
        # Create a base filename without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save a copy of the image
        image_output_path = os.path.join(output_dir, 'images', f"{base_filename}.png")
        image.save(image_output_path)
        
        # Extract text using OCR
        success, text_lines, metadata = extract_text_with_ocr(
            image, name_pattern, date_pattern, location_pattern
        )
        
        if not success:
            return False, f"OCR failed for {image_path}: {metadata.get('error', 'Unknown error')}"
        
        # Add file path to metadata
        metadata['original_file'] = str(image_path)
        
        # Save all extracted text to a file
        text_path = os.path.join(output_dir, 'text', f"{base_filename}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))
        
        # Save structured metadata as JSON
        metadata_path = os.path.join(output_dir, 'metadata', f"{base_filename}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True, base_filename
            
    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"

def find_files(input_dirs, file_types, recursive=False):
    """Find all files of specified types in the given directories"""
    files = []
    for directory in input_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist. Skipping.")
            continue
        
        # Build a pattern for file extensions
        pattern = f"**/*.{{{','.join(file_types)}}}" if recursive else f"*.{{{','.join(file_types)}}}"
        
        # Find matching files
        for file_path in dir_path.glob(pattern):
            files.append(file_path)
    
    return files

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract text from PSD or image files using OCR.')
    parser.add_argument('--input', '-i', required=True, nargs='+', 
                        help='Input directories containing PSD or image files (multiple directories can be specified)')
    parser.add_argument('--output', '-o', required=True, 
                        help='Output directory for extracted data')
    parser.add_argument('--name-pattern', default=r'^[A-Z][a-z]+ [A-Z][a-z]+', 
                        help='Regular expression pattern to identify names')
    parser.add_argument('--date-pattern', default=r'\b\d{4}\b-\b\d{4}\b|\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', 
                        help='Regular expression pattern to identify dates')
    parser.add_argument('--location-pattern', default=r'Section [A-Z0-9]+|Plot [A-Z0-9]+', 
                        help='Regular expression pattern to identify cemetery locations')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search for files recursively in subdirectories')
    parser.add_argument('--file-types', default=['psd', 'png', 'jpg', 'jpeg', 'tif', 'tiff'],
                        nargs='+', help='File extensions to process (default: psd, png, jpg, jpeg, tif, tiff)')
    parser.add_argument('--tesseract-path', 
                        help='Path to Tesseract executable if not in system PATH')
    
    args = parser.parse_args()
    
    # Set Tesseract path if provided
    if args.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    
    # Create output directories
    output_dir = args.output
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    
    # Find all files to process
    files = find_files(args.input, args.file_types, args.recursive)
    print(f"Found {len(files)} files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for index, file_path in enumerate(files):
        print(f"Processing {index+1}/{len(files)}: {file_path}")
        
        file_ext = file_path.suffix.lower()[1:]  # Remove the dot from extension
        
        if file_ext == 'psd':
            # Process as PSD
            success, result = process_psd_file(
                file_path, 
                output_dir, 
                args.name_pattern, 
                args.date_pattern, 
                args.location_pattern
            )
        else:
            # Process as image
            success, result = process_image_file(
                file_path, 
                output_dir, 
                args.name_pattern, 
                args.date_pattern, 
                args.location_pattern
            )
        
        if success:
            successful += 1
        else:
            failed += 1
            print(f"  Error: {result}")
    
    print(f"Processing complete! Successfully processed {successful} files, failed on {failed} files.")
    print(f"Extracted data saved to: {output_dir}")

if __name__ == "__main__":
    main()