import os
import json
import argparse
import re
from psd_tools import PSDImage
from pathlib import Path

def extract_text_from_layers(layers, metadata, name_pattern, date_pattern, location_pattern):
    """Recursively extract text from PSD layers and categorize it"""
    text_found = []
    for layer in layers:
        if hasattr(layer, 'text_data') and layer.text_data:
            # This is a text layer
            text = layer.text_data.text
            text_found.append(text)
            
            # Try to categorize the text
            if re.match(name_pattern, text):
                metadata['name'] = text
            elif re.search(date_pattern, text):
                dates = re.findall(date_pattern, text)
                metadata['dates'].extend(dates)
            elif re.search(location_pattern, text):
                metadata['location'] = text
            else:
                metadata['notes'].append(text)
                
        # If it's a group with nested layers, process those too
        if hasattr(layer, 'layers') and layer.layers:
            nested_text = extract_text_from_layers(layer.layers, metadata, 
                                                  name_pattern, date_pattern, location_pattern)
            text_found.extend(nested_text)
    return text_found

def process_psd_file(psd_path, output_dir, name_pattern, date_pattern, location_pattern):
    """Process a single PSD file and extract its content"""
    try:
        # Open the PSD file
        psd = PSDImage.open(psd_path)
        
        # Create a base filename without extension
        base_filename = os.path.splitext(os.path.basename(psd_path))[0]
        
        # Extract and save the composite image
        image = psd.composite()
        image_path = os.path.join(output_dir, 'images', f"{base_filename}.png")
        image.save(image_path)
        
        # Extract text from all text layers
        all_text = []
        metadata = {
            'original_file': str(psd_path),
            'name': '',
            'dates': [],
            'location': '',
            'notes': []
        }
        
        all_text = extract_text_from_layers(psd.layers, metadata, 
                                           name_pattern, date_pattern, location_pattern)
        
        # Save all extracted text to a file
        text_path = os.path.join(output_dir, 'text', f"{base_filename}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        # Save structured metadata as JSON
        metadata_path = os.path.join(output_dir, 'metadata', f"{base_filename}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True, base_filename
            
    except Exception as e:
        return False, f"Error processing {psd_path}: {str(e)}"

def find_psd_files(input_dirs, recursive=False):
    """Find all PSD files in the specified directories"""
    psd_files = []
    for directory in input_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist. Skipping.")
            continue
            
        # Find all PSD files in this directory
        if recursive:
            for file_path in dir_path.glob('**/*.psd'):
                psd_files.append(file_path)
        else:
            for file_path in dir_path.glob('*.psd'):
                psd_files.append(file_path)
    
    return psd_files

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Extract content from PSD files containing gravestone images.')
    parser.add_argument('--input', '-i', required=True, nargs='+', 
                        help='Input directories containing PSD files (multiple directories can be specified)')
    parser.add_argument('--output', '-o', required=True, 
                        help='Output directory for extracted data')
    parser.add_argument('--name-pattern', default=r'^[A-Z][a-z]+ [A-Z][a-z]+', 
                        help='Regular expression pattern to identify names')
    parser.add_argument('--date-pattern', default=r'\b\d{4}\b-\b\d{4}\b|\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', 
                        help='Regular expression pattern to identify dates')
    parser.add_argument('--location-pattern', default=r'Section [A-Z0-9]+|Plot [A-Z0-9]+', 
                        help='Regular expression pattern to identify cemetery locations')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search for PSD files recursively in subdirectories')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = args.output
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    
    # Find all PSD files in the specified directories
    psd_files = find_psd_files(args.input, args.recursive)
    print(f"Found {len(psd_files)} PSD files to process")
    
    # Process each PSD file
    successful = 0
    failed = 0
    
    for index, psd_path in enumerate(psd_files):
        print(f"Processing {index+1}/{len(psd_files)}: {psd_path}")
        success, result = process_psd_file(
            psd_path, 
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