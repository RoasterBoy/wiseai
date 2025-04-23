import os
import json
import argparse
import re
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from psd_tools import PSDImage
from pathlib import Path

def enhance_image_for_ocr(image):
    """Apply image enhancements to improve OCR results"""
    # Convert to grayscale
    gray_image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(gray_image)
    contrast_image = enhancer.enhance(2.0)  # Adjust contrast factor as needed
    
    # Apply sharpening
    sharp_image = contrast_image.filter(ImageFilter.SHARPEN)
    
    return sharp_image

def extract_text_with_ocr_debug(image, output_dir, base_filename, name_pattern, date_pattern, location_pattern):
    """Extract text from image using OCR with debug output"""
    # Save the original image for reference
    original_path = os.path.join(output_dir, 'debug', f"{base_filename}_original.png")
    image.save(original_path)
    
    # Try OCR on the original image
    print(f"Running OCR on original image...")
    original_text = pytesseract.image_to_string(image)
    
    with open(os.path.join(output_dir, 'debug', f"{base_filename}_original_ocr.txt"), 'w', encoding='utf-8') as f:
        f.write(original_text)
    
    print(f"OCR text length: {len(original_text)} characters")
    print(f"Sample text (first 150 chars): {original_text[:150]}")
    
    # Try with image enhancement
    print(f"Enhancing image for OCR...")
    enhanced_image = enhance_image_for_ocr(image)
    enhanced_path = os.path.join(output_dir, 'debug', f"{base_filename}_enhanced.png")
    enhanced_image.save(enhanced_path)
    
    enhanced_text = pytesseract.image_to_string(enhanced_image)
    
    with open(os.path.join(output_dir, 'debug', f"{base_filename}_enhanced_ocr.txt"), 'w', encoding='utf-8') as f:
        f.write(enhanced_text)
    
    print(f"Enhanced OCR text length: {len(enhanced_text)} characters")
    print(f"Enhanced sample text (first 150 chars): {enhanced_text[:150]}")
    
    # Try with different OCR configurations
    print(f"Trying OCR with different configurations...")
    
    # Try with tesseract page segmentation modes
    psm_modes = [3, 4, 6, 11, 12]  # Different page segmentation modes to try
    best_text = ""
    best_length = 0
    best_psm = 0
    
    for psm in psm_modes:
        config = f'--psm {psm}'
        mode_text = pytesseract.image_to_string(enhanced_image, config=config)
        
        with open(os.path.join(output_dir, 'debug', f"{base_filename}_psm{psm}_ocr.txt"), 'w', encoding='utf-8') as f:
            f.write(mode_text)
        
        print(f"PSM {psm} OCR text length: {len(mode_text)} characters")
        
        # Keep track of which mode gave the most text
        if len(mode_text) > best_length:
            best_length = len(mode_text)
            best_text = mode_text
            best_psm = psm
    
    print(f"Best OCR results from PSM {best_psm} with {best_length} characters")
    
    # Use the best text for further processing
    if best_length > len(enhanced_text):
        print(f"Using PSM {best_psm} results")
        text_to_use = best_text
    else:
        print("Using enhanced image results")
        text_to_use = enhanced_text
    
    # Split by newlines to get separate text elements
    text_lines = [line.strip() for line in text_to_use.split('\n') if line.strip()]
    
    # Initialize metadata
    metadata = {
        'name': '',
        'dates': [],
        'location': '',
        'notes': []
    }
    
    # Categorize each line of text
    print("\nCategorizing text lines:")
    for line in text_lines:
        print(f"Line: '{line}'")
        
        if re.match(name_pattern, line):
            print(f"  Matched as NAME")
            metadata['name'] = line
        elif re.search(date_pattern, line):
            print(f"  Matched as DATE")
            dates = re.findall(date_pattern, line)
            print(f"  Found dates: {dates}")
            metadata['dates'].extend(dates)
        elif re.search(location_pattern, line):
            print(f"  Matched as LOCATION")
            metadata['location'] = line
        else:
            print(f"  Added to NOTES")
            metadata['notes'].append(line)
    
    # Print regex patterns for reference
    print(f"\nRegex patterns used:")
    print(f"  Name pattern: {name_pattern}")
    print(f"  Date pattern: {date_pattern}")
    print(f"  Location pattern: {location_pattern}")
    
    return True, text_lines, metadata, text_to_use

def process_image_file_debug(image_path, output_dir, name_pattern, date_pattern, location_pattern):
    """Process a single image file with debugging output"""
    try:
        print(f"\n=== Processing {image_path} ===")
        
        # Create debug directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)
        
        # Open the image file
        image = Image.open(image_path)
        print(f"Image opened: {image.format}, {image.size}, {image.mode}")
        
        # Create a base filename without extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save a copy of the image
        image_output_path = os.path.join(output_dir, 'images', f"{base_filename}.png")
        image.save(image_output_path)
        
        # Extract text using OCR with debugging
        success, text_lines, metadata, raw_text = extract_text_with_ocr_debug(
            image, output_dir, base_filename, name_pattern, date_pattern, location_pattern
        )
        
        # Add file path to metadata
        metadata['original_file'] = str(image_path)
        
        # Save all extracted text to a file
        text_path = os.path.join(output_dir, 'text', f"{base_filename}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
        # Save structured metadata as JSON
        metadata_path = os.path.join(output_dir, 'metadata', f"{base_filename}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True, base_filename
            
    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Debug OCR text extraction from image files.')
    parser.add_argument('--input', '-i', required=True, 
                        help='Input file to process')
    parser.add_argument('--output', '-o', required=True, 
                        help='Output directory for extracted data')
    parser.add_argument('--name-pattern', default=r'^[A-Z][a-z]+ [A-Z][a-z]+', 
                        help='Regular expression pattern to identify names')
    parser.add_argument('--date-pattern', default=r'\b\d{4}\b-\b\d{4}\b|\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', 
                        help='Regular expression pattern to identify dates')
    parser.add_argument('--location-pattern', default=r'Section [A-Z0-9]+|Plot [A-Z0-9]+', 
                        help='Regular expression pattern to identify cemetery locations')
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
    os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)
    
    # Process the file
    file_path = Path(args.input)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return
    
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    success, result = process_image_file_debug(
        file_path, 
        output_dir, 
        args.name_pattern, 
        args.date_pattern, 
        args.location_pattern
    )
    
    if success:
        print(f"\nProcessing complete! Check the debug folder for OCR result details.")
    else:
        print(f"\nError: {result}")

if __name__ == "__main__":
    main()