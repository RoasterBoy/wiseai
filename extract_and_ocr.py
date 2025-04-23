# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import argparse
import pytesseract
import pandas as pd
import json
from pytesseract import Output
import traceback # For detailed error printing
import re # **** IMPORT re MODULE ****


# --- Configuration ---
# Set the path to the Tesseract executable if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Helper Functions ---

# REVISED FUNCTION to use confidence threshold argument
def get_text_blocks(ocr_df, confidence_threshold=0): # Defaulting to 0 for max inclusion
    """Groups words from OCR data into text blocks based on paragraph and block numbers."""

    # --- Step 1: Initial Validation and Type Conversion ---
    if ocr_df.empty: return [] # Return early if input is empty

    # Ensure essential columns exist and are numeric before filtering
    required_numeric_cols = ['conf', 'left', 'top', 'width', 'height', 'block_num', 'par_num', 'line_num', 'word_num']
    for col in required_numeric_cols:
        if col in ocr_df.columns:
            # Use errors='coerce' to turn non-numeric values into NaN
            ocr_df[col] = pd.to_numeric(ocr_df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' missing in raw OCR data.")
            # If essential geometry/grouping columns are missing, we probably can't proceed
            if col in ['left', 'top', 'width', 'height', 'block_num', 'par_num', 'line_num']:
                 print("Cannot reliably process blocks. Returning empty list.")
                 return []

    # Ensure 'text' column exists and is string
    if 'text' in ocr_df.columns:
        ocr_df['text'] = ocr_df['text'].fillna('').astype(str)
    else:
        print("Warning: 'text' column missing in raw OCR data. Returning empty list.")
        return []

    # Drop rows where essential numeric conversions failed (resulted in NaN)
    ocr_df.dropna(subset=['conf', 'left', 'top', 'width', 'height', 'block_num', 'par_num', 'line_num'], inplace=True)

    # --- Step 2: Confidence Filtering and Copying ---
    # Use .loc to filter and copy to reliably avoid SettingWithCopyWarning
    # Filter out negative confidence values which Tesseract uses for non-text elements
    # Also filter based on the provided threshold
    filtered_df = ocr_df.loc[(ocr_df['conf'] >= confidence_threshold) & (ocr_df['conf'] >= 0)].copy()

    if filtered_df.empty:
        # print(f"--- DEBUG: DataFrame empty after confidence filter (>{confidence_threshold}) ---") # Optional debug
        return []

    # --- Step 3: Process the Copy ---
    # Text already converted to string earlier
    filtered_df['text'] = filtered_df['text'].str.strip()
    # Remove rows with empty text that might have resulted from stripping whitespace-only strings
    filtered_df = filtered_df[filtered_df['text'] != '']

    if filtered_df.empty: # Check again after stripping text
        return []

    # Sort the copy - Ensure columns are integer type before sorting
    sort_cols = ['block_num', 'par_num', 'line_num', 'word_num']
    # Check existence again on filtered_df as columns could be all NaN and dropped
    if all(col in filtered_df.columns for col in sort_cols):
         try:
             # Convert sort columns to integer type for reliable sorting
             filtered_df[sort_cols] = filtered_df[sort_cols].astype(int)
             filtered_df.sort_values(by=sort_cols, inplace=True)
         except Exception as sort_err:
              print(f"--- WARNING: Error during sorting: {sort_err} ---")
              # Attempt to continue without sorting if error occurs

    # --- Step 4: Grouping and Line Creation ---
    lines_data = []
    try:
        group_cols = ['block_num', 'par_num', 'line_num']
        if all(col in filtered_df.columns for col in group_cols):
            # Ensure group cols are integer type before grouping
            filtered_df[group_cols] = filtered_df[group_cols].astype(int)
            # Group by block, paragraph, and line number
            grouped = filtered_df.groupby(group_cols, sort=False) # Use sort=False as data is pre-sorted

            required_cols_group = ['left', 'top', 'width', 'height', 'text', 'conf']
            for name, group in grouped:
                # Ensure the group is not empty and contains required columns
                if group.empty or not all(col in group.columns for col in required_cols_group):
                    continue
                # Drop rows within the group if essential numeric conversion failed earlier (resulted in NaN)
                group.dropna(subset=['left', 'top', 'width', 'height', 'conf'], inplace=True)
                if group.empty: continue

                # Join the text of words in the line
                line_text = ' '.join(group['text'].tolist())
                if line_text: # Check if line_text is not empty after joining
                    try:
                        # Calculate bounding box for the entire line
                        first_word = group.iloc[0] # Get first word for block/par/line numbers
                        x = group['left'].min()
                        y = group['top'].min()
                        # Calculate max right edge and max bottom edge for width/height
                        w = (group['left'] + group['width']).max() - x
                        h = (group['top'] + group['height']).max() - y

                        # Ensure dimensions are non-negative integers
                        x, y, w, h = int(x), int(y), int(max(0, w)), int(max(0, h))

                        # Get block, paragraph, line numbers safely
                        block_num = int(first_word.get('block_num', 0))
                        par_num = int(first_word.get('par_num', 0))
                        line_num = int(first_word.get('line_num', 0))

                        # Calculate average confidence for the line
                        avg_conf = float(group['conf'].mean())

                        lines_data.append({
                            'block_num': block_num, 'par_num': par_num, 'line_num': line_num,
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'text': line_text, 'conf': avg_conf
                        })
                    except Exception as e:
                        print(f"Warning: Could not process OCR group {name}. Error: {e}")
                        traceback.print_exc() # Print detailed error

    except Exception as group_err:
        print(f"--- WARNING: Error during grouping: {group_err} ---")
        traceback.print_exc() # Print detailed error

    return lines_data


# SIMPLIFIED FUNCTION - Focus on Header/Footer, ignore Location/Add. Info for now
def classify_text_block(block, index, all_blocks, img_width, img_height):
    """Classifies text block based on location, focusing on Header/Footer."""
    # Ensure block has coordinate keys before proceeding
    if not all(k in block for k in ('x', 'y', 'w', 'h')):
        # Assign default type and usage flag if coordinates are missing
        block['type'] = "Body/Caption"
        block['used_as_caption'] = False
        block['initial_type'] = "Body/Caption" # Add for consistency
        return "Body/Caption"

    x, y, w, h = block['x'], block['y'], block['w'], block['h']
    center_x = x + w / 2; block_y = y

    # **** ZONE DEFINITIONS ****
    # Keep zone definitions for clarity, but logic using them will be simplified
    header_zone_y_limit = img_height * 0.25  # Top 25%
    footer_zone_y_start = img_height * 0.80  # Bottom 20%
    center_zone_x_min = img_width * 0.25; center_zone_x_max = img_width * 0.75
    # **** END OF ZONE DEFINITIONS ****

    final_type = "Body/Caption" # Default type

    # --- Simplified Classification ---
    if block_y < header_zone_y_limit: # In the top zone
        # Prioritize Header if block is horizontally centered
        if center_zone_x_min < center_x < center_zone_x_max:
             final_type = "Header"
        # Text in top-left or top-right will likely remain Body/Caption
        else:
            # Optional: Fallback check if it starts near center - less certain header
            if center_zone_x_min < x < center_zone_x_max:
                 final_type = "Potential Header/Info"
            else:
                 final_type = "Body/Caption" # Default for far left/right top zone

    elif block_y > footer_zone_y_start: # In the bottom zone
        # Prioritize Footer if block is horizontally centered
        if center_zone_x_min < center_x < center_zone_x_max:
            final_type = "Footer"
        else: # Off-center in the bottom zone
             final_type = "Potential Footer"

    # Assign the final determined type to the current block
    block['type'] = final_type
    # Mark the block as used if it's a structural element
    if final_type in ["Header", "Footer", "Potential Header/Info", "Potential Footer"]:
        block['used_as_caption'] = True
    # Ensure 'used_as_caption' exists, default to False if not set above
    if 'used_as_caption' not in block:
        block['used_as_caption'] = False

    # Store initial type just for potential debugging, not used in logic now
    block['initial_type'] = final_type

    # No complex contextual refinement needed in this simplified version
    return final_type


# Unchanged find_caption_for_image function
def find_caption_for_image(image_bbox, text_blocks, max_distance_y=100, max_distance_x=150):
    """Finds the most likely caption for an image based on proximity."""
    img_x, img_y, img_w, img_h = image_bbox
    img_bottom = img_y + img_h; img_right = img_x + img_w; img_center_x = img_x + img_w / 2
    best_caption = ""; min_dist = float('inf'); best_block_idx = -1
    for i, block in enumerate(text_blocks):
        if not all(k in block for k in ('x', 'y', 'w', 'h')): continue
        # Skip if block is already used as a structural element or assigned as caption
        if block.get('used_as_caption'): continue

        txt_x, txt_y, txt_w, txt_h = block['x'], block['y'], block['w'], block['h']; txt_center_x = txt_x + txt_w / 2

        # Check below the image first
        if txt_y >= img_bottom - 10: # Allow slight overlap upwards
             vertical_dist = txt_y - img_bottom;
             overlap_x = max(0, min(img_x + img_w, txt_x + txt_w) - max(img_x, txt_x));
             center_dist_x = abs(txt_center_x - img_center_x)

             if -10 <= vertical_dist < max_distance_y and (overlap_x > img_w * 0.2 or center_dist_x < img_w * 0.5):
                 dist = max(0, vertical_dist) + center_dist_x * 0.2
                 if dist < min_dist: min_dist = dist; best_caption = block['text']; best_block_idx = i

        # Check to the right of the image ONLY if no suitable caption found below
        elif txt_x >= img_right - 10 and best_block_idx == -1: # Allow slight overlap leftwards
             horizontal_dist = txt_x - img_right;
             overlap_y = max(0, min(img_y + img_h, txt_y + txt_h) - max(img_y, txt_y));
             center_dist_y = abs((txt_y + txt_h/2) - (img_y + img_h/2))

             if -10 <= horizontal_dist < max_distance_x and (overlap_y > img_h * 0.2 or center_dist_y < img_h * 0.5):
                  dist = max(0, horizontal_dist) + center_dist_y * 0.2
                  if dist < min_dist: min_dist = dist; best_caption = block['text']; best_block_idx = i

    # Mark the best block as used so it's not reused / misclassified later
    if best_block_idx != -1 and best_block_idx < len(text_blocks):
        # Double check it's not a structural block before marking
        if text_blocks[best_block_idx].get('type') not in ["Header", "Footer", "Location", "Additional Info", "Potential Header/Info", "Potential Footer"]:
             text_blocks[best_block_idx]['used_as_caption'] = True

    return best_caption


# --- Main Processing Function ---
def extract_images_and_text(image_path, output_dir, config):
    """Extracts images and text, associating text with images."""
    results = []; base_filename = os.path.splitext(os.path.basename(image_path))[0]
    try:
        img_original = cv2.imread(image_path) # Load original image
        if img_original is None: print(f"Error: Could not read image {image_path}"); return results
        img_height, img_width = img_original.shape[:2]; total_image_area = img_height * img_width
        print(f"\nProcessing {image_path} ({img_width}x{img_height})")

        # --- Pre-processing Step: Otsu's Thresholding ---
        print("  Applying Otsu's thresholding for OCR...") # Indicate method change
        img_for_ocr = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's Binarization
        # The threshold value (ret_val) is calculated automatically by Otsu's method,
        # but we don't need it directly, only the thresholded image (img_for_ocr).
        ret_val, img_for_ocr = cv2.threshold(img_for_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # **** Option to save the thresholded image for debugging ****
        # debug_threshold_filename = os.path.join(output_dir, f"{base_filename}_otsu_debug.png")
        # print(f"  Saving Otsu debug image to: {debug_threshold_filename}")
        # cv2.imwrite(debug_threshold_filename, img_for_ocr)
        # **** End debug save option ****

        img_to_process = img_for_ocr # Use the Otsu-thresholded image
        # --- End Pre-processing Step ---

        print("  Performing OCR...")
        try:
             # Pass PSM config
             tess_config = f'--psm {config["psm"]}'
             # Pass the Otsu-thresholded image
             ocr_data = pytesseract.image_to_data(img_to_process, output_type=Output.DATAFRAME, lang='eng', config=tess_config)

             if not isinstance(ocr_data, pd.DataFrame) or ocr_data.empty:
                 print(f"Warning: OCR for {image_path} returned empty or non-DataFrame result.")
                 ocr_data = pd.DataFrame()

             print(f"  OCR initial result rows: {len(ocr_data)}")
        except pytesseract.TesseractError as ocr_err: print(f"  Tesseract Error during OCR for {image_path}: {ocr_err}"); ocr_data = pd.DataFrame()
        except Exception as e: print(f"  Unexpected Error during OCR setup/execution for {image_path}: {e}"); ocr_data = pd.DataFrame()

        # Pass min_conf config
        text_blocks = get_text_blocks(ocr_data, config['min_conf']) # Pass raw (but checked type) DF and confidence
        print(f"  Grouped into {len(text_blocks)} text lines/blocks (conf >= {config['min_conf']}).")


        # --- Classification and Debugging ---
        # This loop uses the simplified classify_text_block
        print("  --- Debugging Text Block Classification ---")
        for i, block in enumerate(text_blocks):
            # Classification logic is now simplified
            block_type = classify_text_block(block, i, text_blocks, img_width, img_height)
            # Debug print shows the final type assigned by the simplified logic
            block_coords = f"x:{block.get('x', 'N/A')}, y:{block.get('y', 'N/A')}, w:{block.get('w', 'N/A')}, h:{block.get('h', 'N/A')}"
            print(f"    Block {i}: Final Type='{block.get('type', 'N/A')}', Coords=[{block_coords}], Text='{block.get('text', '')[:50]}...'") # Print first 50 chars
        print("  --- End Debugging ---")


        # --- Aggregate Classified Blocks ---
        # This pass aggregates based on the types assigned by the simplified logic
        header_text = []; footer_text = []; location_text = []; additional_info_text = []
        for block in text_blocks:
            block_type = block.get('type', 'Body/Caption')
            # Aggregate based on the simplified types
            if block_type == "Header": header_text.append(block.get('text',''))
            elif block_type == "Footer": footer_text.append(block.get('text',''))
            # Location and Additional Info are no longer actively classified

        # Join the aggregated text for final metadata
        raw_header = " | ".join(filter(None, header_text))
        raw_footer = " | ".join(filter(None, footer_text))
        raw_location = " ".join(filter(None, location_text))
        raw_additional_info = " | ".join(filter(None, additional_info_text))

    # **** REFINED HEADER CLEANING STEP ****
    # 1. Take only the part before the first " | " delimiter (if exists) and strip whitespace
        cleaned_header = raw_header.split('|')[0].strip() if raw_header else ""
    # 2. Use regex to remove space followed by digits ONLY at the very end ($) of the string
        cleaned_header = re.sub(r'\s+\d+$', '', cleaned_header)
    # **** END OF REFINED CLEANING STEP ****

        page_metadata = {
            "Header": cleaned_header, # Assign the cleaned header
            "Footer": raw_footer, # Keep raw footer for now
            "Location": raw_location, # Expect empty
            "Additional Info": raw_additional_info # Expect empty
        }
        print(f"  Page Metadata: {page_metadata}") # Check this line for the cleaned header


        # --- Image Detection and Captioning ---
        # Image detection uses the original image
        print("  Detecting image regions...")
        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY); blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use threshold value from config for image detection
        _, thresh = cv2.threshold(blurred, config['threshold_value'], 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_bounding_boxes = []
        if contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour); min_pixel_width = 20; min_pixel_height = 20
                if w < min_pixel_width or h < min_pixel_height: continue
                area = w * h;
                if area == 0: continue
                aspect_ratio = float(w) / h if h > 0 else 0; area_ratio = area / total_image_area
                # Use detection parameters from config
                if (config['min_area_ratio'] < area_ratio < config['max_area_ratio'] and
                    config['aspect_ratio_min'] < aspect_ratio < config['aspect_ratio_max']):
                    valid_bounding_boxes.append((x, y, w, h))
        # Sort detected boxes top-to-bottom, then left-to-right
        valid_bounding_boxes.sort(key=lambda box: (box[1], box[0]))
        print(f"  Found {len(valid_bounding_boxes)} potential image regions.")

        # --- Process and Save Extracted Images ---
        os.makedirs(output_dir, exist_ok=True); extracted_count = 0
        for i, (x, y, w, h) in enumerate(valid_bounding_boxes):
            image_bbox = (x, y, w, h)
            # Find caption using the current state of text_blocks
            caption = find_caption_for_image(image_bbox, text_blocks,
                                             max_distance_y=config['caption_max_dist_y'],
                                             max_distance_x=config['caption_max_dist_x'])
            # Apply padding from config
            padding = config['padding']
            y1 = max(0, y - padding); y2 = min(img_height, y + h + padding)
            x1 = max(0, x - padding); x2 = min(img_width, x + w + padding)

            # Ensure crop dimensions are valid
            if y1 >= y2 or x1 >= x2:
                print(f"    Warning: Invalid crop dimensions for image {i+1} after padding. Skipping save.")
                continue

            cropped_img = img_original[y1:y2, x1:x2] # Crop from original

            if cropped_img is None or cropped_img.size == 0:
                print(f"    Warning: Cropped image {i+1} is empty. Skipping save.")
                continue

            output_filename = os.path.join(output_dir, f"{base_filename}_img_{i+1:02d}.jpg")
            try:
                success = cv2.imwrite(output_filename, cropped_img)
                if not success:
                    print(f"    Error: OpenCV failed to save image {output_filename}.")
                    continue
                # Append metadata for the successfully saved image
                results.append({
                    "source_page": image_path,
                    "extracted_image": output_filename,
                    "image_bounding_box": [int(x), int(y), int(w), int(h)],
                    "page_header": page_metadata["Header"], # Use cleaned header
                    "page_footer": page_metadata["Footer"],
                    "page_location": page_metadata["Location"],
                    "page_additional_info": page_metadata["Additional Info"],
                    "image_caption": caption
                })
                extracted_count += 1
            except Exception as e:
                print(f"    Error during image save or metadata append for {output_filename}: {e}")
                traceback.print_exc() # Print detailed error

        print(f"  Successfully processed and potentially saved {extracted_count} images from {image_path}")
        return results
    except Exception as e:
        print(f"An unexpected major error occurred processing {image_path}: {e}")
        traceback.print_exc(); # Print detailed error for major failures
        return []

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract individual images and associated text from scanned pages.")
    parser.add_argument("input_folder", help="Folder containing the scanned image files.")
    parser.add_argument("output_folder", help="Folder where extracted images and metadata JSON will be saved.")
    # Image detection parameters
    parser.add_argument("--min_area", type=float, default=0.005, help="Minimum image area as a fraction of total page area.")
    parser.add_argument("--max_area", type=float, default=0.4, help="Maximum image area as a fraction of total page area.")
    parser.add_argument("--min_aspect", type=float, default=0.3, help="Minimum aspect ratio (width/height) for detected images.")
    parser.add_argument("--max_aspect", type=float, default=3.0, help="Maximum aspect ratio (width/height) for detected images.")
    parser.add_argument("--padding", type=int, default=5, help="Padding pixels around detected images before cropping.")
    parser.add_argument("--threshold", type=int, default=235, help="Threshold value for image detection (used in cv2.threshold).")
    # Captioning parameters
    parser.add_argument("--caption_max_dist_y", type=int, default=75, help="Max vertical distance below an image to search for a caption.")
    parser.add_argument("--caption_max_dist_x", type=int, default=100, help="Max horizontal distance right of an image to search for a caption.")
    # OCR parameters
    parser.add_argument("--min_conf", type=int, default=0, help="Minimum OCR confidence level (0-100) to keep a word. Default 0 includes all.")
    parser.add_argument("--psm", type=int, default=3, help="Tesseract Page Segmentation Mode (0-13). Default 3.")
    # General parameters
    parser.add_argument('--ext', nargs='+', default=['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'], help="List of file extensions to process.")
    parser.add_argument('--output_json', default="metadata.json", help="Name for the output metadata JSON file.")
    parser.add_argument('--tesseract_cmd', default=None, help="Explicit path to the Tesseract executable if not in PATH.")
    args = parser.parse_args()

    # --- Tesseract Check ---
    if args.tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    try: version_info = pytesseract.get_tesseract_version(); print(f"Using Tesseract OCR version: {version_info}")
    except pytesseract.TesseractNotFoundError: print(f"\nError: Tesseract executable not found.\nAttempted path: {pytesseract.pytesseract.tesseract_cmd}\nPlease install Tesseract OCR and ensure it's in your system's PATH,\nor provide the correct path using the --tesseract_cmd argument."); exit(1)
    except Exception as e: print(f"\nError accessing Tesseract: {e}\nThere might be an issue with your Tesseract installation or the provided path."); exit(1)

    valid_extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in args.ext]
    all_metadata = []; processed_files = 0
    # Pass all parameters into the config dict
    config = {
        'min_area_ratio': args.min_area, 'max_area_ratio': args.max_area,
        'aspect_ratio_min': args.min_aspect, 'aspect_ratio_max': args.max_aspect,
        'padding': args.padding, 'threshold_value': args.threshold,
        'caption_max_dist_y': args.caption_max_dist_y, 'caption_max_dist_x': args.caption_max_dist_x,
        'min_conf': args.min_conf,
        'psm': args.psm
    }
    print(f"\nScanning folder: {args.input_folder}")
    print(f"Outputting images to: {args.output_folder}")
    print(f"Outputting metadata to: {os.path.join(args.output_folder, args.output_json)}")
    print(f"Processing extensions: {valid_extensions}")
    print(f"Image Detection Threshold: {args.threshold}")
    print(f"OCR Confidence Threshold: >= {args.min_conf}")
    print(f"OCR Page Segmentation Mode: {args.psm}")
    print("OCR Pre-processing: Otsu's Thresholding") # Indicate pre-processing method
    print("-" * 30)
    os.makedirs(args.output_folder, exist_ok=True)

    # --- Main Loop ---
    for filename in sorted(os.listdir(args.input_folder)):
        input_path = os.path.join(args.input_folder, filename)
        if os.path.isfile(input_path) and os.path.splitext(filename)[1].lower() in valid_extensions:
            processed_files += 1
            page_results = extract_images_and_text(input_path, args.output_folder, config)
            if page_results: all_metadata.extend(page_results)
            print("-" * 30)
        else: pass # Skip directories or non-matching files silently

    # --- Save Metadata ---
    output_json_path = os.path.join(args.output_folder, args.output_json)
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f: json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\nMetadata saved to {output_json_path}")
    except TypeError as te: print(f"\nError saving metadata JSON: Data might not be JSON serializable. Error: {te}")
    except Exception as e: print(f"\nError saving metadata JSON: {e}")

    total_extracted = len(all_metadata)
    print(f"Batch processing complete. Processed {processed_files} files. Extracted {total_extracted} images with metadata.")