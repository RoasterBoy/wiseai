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

# --- Configuration ---
# Set the path to the Tesseract executable if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Helper Functions ---

# REVISED FUNCTION to fix SettingWithCopyWarning and LOWER confidence threshold
def get_text_blocks(ocr_df):
    """Groups words from OCR data into text blocks based on paragraph and block numbers."""
    
    # --- Step 1: Initial Validation and Type Conversion ---
    if ocr_df.empty: return [] # Return early if input is empty
        
    # Ensure essential columns exist and are numeric before filtering
    required_numeric_cols = ['conf', 'left', 'top', 'width', 'height', 'block_num', 'par_num', 'line_num', 'word_num']
    for col in required_numeric_cols:
        if col in ocr_df.columns:
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
        
    # Drop rows where essential numeric conversions failed
    ocr_df.dropna(subset=['conf', 'left', 'top', 'width', 'height', 'block_num', 'par_num', 'line_num'], inplace=True) 
    
    # --- Step 2: Confidence Filtering and Copying ---
    # **** LOWERED CONFIDENCE THRESHOLD ****
    confidence_threshold = 10 
    # Use .loc to filter and copy to reliably avoid SettingWithCopyWarning
    filtered_df = ocr_df.loc[ocr_df['conf'] > confidence_threshold].copy() 
    # **** END OF CHANGE ****

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

    # Sort the copy - Ensure columns are integer type
    sort_cols = ['block_num', 'par_num', 'line_num', 'word_num']
    # Check existence again on filtered_df as columns could be all NaN and dropped
    if all(col in filtered_df.columns for col in sort_cols):
         try: 
             filtered_df[sort_cols] = filtered_df[sort_cols].astype(int) # Convert directly now
             filtered_df.sort_values(by=sort_cols, inplace=True)
         except Exception as sort_err:
              print(f"--- WARNING: Error during sorting: {sort_err} ---")
    # else: # Removed noisy warning


    # --- Step 4: Grouping and Line Creation ---
    lines_data = []
    try: 
        group_cols = ['block_num', 'par_num', 'line_num']
        if all(col in filtered_df.columns for col in group_cols):
            # Ensure group cols are integer type after potential fillna(0)
            filtered_df[group_cols] = filtered_df[group_cols].astype(int)
            grouped = filtered_df.groupby(group_cols, sort=False) # Use sort=False as data is pre-sorted
            
            required_cols_group = ['left', 'top', 'width', 'height', 'text', 'conf']
            for name, group in grouped:
                if group.empty or not all(col in group.columns for col in required_cols_group):
                    continue
                # Drop rows within group if essential numeric conversion failed earlier
                group.dropna(subset=['left', 'top', 'width', 'height', 'conf'], inplace=True)
                if group.empty: continue

                line_text = ' '.join(group['text'].tolist())
                if line_text: # Check if not empty after join
                    try:
                        first_word = group.iloc[0]
                        x = group['left'].min(); y = group['top'].min()
                        w = group['left'].max() + group['width'].max() - x 
                        h = group['top'].max() + group['height'].max() - y
                        block_num = int(first_word.get('block_num', 0)); par_num = int(first_word.get('par_num', 0)); line_num = int(first_word.get('line_num', 0))
                        lines_data.append({'block_num': block_num, 'par_num': par_num, 'line_num': line_num,'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),'text': line_text, 'conf': float(group['conf'].mean()) })
                    except Exception as e: print(f"Warning: Could not process OCR group {name}. Error: {e}")
        # else: # Removed noisy warning
            
    except Exception as group_err:
        print(f"--- WARNING: Error during grouping: {group_err} ---")

    return lines_data

# Uses increased header_zone_y_limit
def classify_text_block(block, index, all_blocks, img_width, img_height):
    """Classifies text block based on location and context."""
    if not all(k in block for k in ('x', 'y', 'w', 'h')): return "Body/Caption" 
    x, y, w, h = block['x'], block['y'], block['w'], block['h']
    center_x = x + w / 2; block_y = y 
    header_zone_y_limit = img_height * 0.25  # Top 25%
    footer_zone_y_start = img_height * 0.80  
    center_zone_x_min = img_width * 0.25; center_zone_x_max = img_width * 0.75
    right_zone_x_start = img_width * 0.65; left_zone_x_limit = img_width * 0.30
    initial_type = "Body/Caption" 
    if block_y < header_zone_y_limit: 
        if center_x > right_zone_x_start:
            if x > img_width * 0.5: initial_type = "Location"
        elif center_zone_x_min < center_x < center_zone_x_max: initial_type = "Header"
        elif center_x < left_zone_x_limit: initial_type = "Additional Info"
        else: # Fallback within top zone
            if center_zone_x_min < x < center_zone_x_max: initial_type = "Header" 
            elif x > center_zone_x_max: initial_type = "Location" 
            else: initial_type = "Potential Header/Info"
    elif block_y > footer_zone_y_start: 
        if center_zone_x_min < center_x < center_zone_x_max: initial_type = "Footer"
        else: initial_type = "Potential Footer"
    # Contextual Refinement (Location)
    if initial_type != "Location" and index > 0:
        if index -1 < len(all_blocks):
             prev_block = all_blocks[index - 1]
             if prev_block.get('type') == "Location" and all(k in prev_block for k in ('x', 'y', 'w', 'h')): 
                prev_x, prev_y, prev_w, prev_h = prev_block['x'], prev_block['y'], prev_block['w'], prev_block['h']
                vertical_gap = y - (prev_y + prev_h); horizontal_diff = abs(center_x - (prev_x + prev_w / 2))
                if 0 <= vertical_gap < (prev_h * 2.0) and horizontal_diff < prev_w and x > img_width * 0.5: initial_type = "Location" 
    return initial_type

# Unchanged find_caption_for_image function
def find_caption_for_image(image_bbox, text_blocks, max_distance_y=100, max_distance_x=150):
    """Finds the most likely caption for an image based on proximity."""
    # ... (function remains the same as previous version) ...
    img_x, img_y, img_w, img_h = image_bbox
    img_bottom = img_y + img_h; img_right = img_x + img_w; img_center_x = img_x + img_w / 2
    best_caption = ""; min_dist = float('inf'); best_block_idx = -1
    for i, block in enumerate(text_blocks):
        if not all(k in block for k in ('x', 'y', 'w', 'h')): continue
        if block.get('used_as_caption'): continue
        if block.get('type', 'Body/Caption') not in ["Body/Caption", "Potential Header/Info", "Potential Footer"]: continue
        txt_x, txt_y, txt_w, txt_h = block['x'], block['y'], block['w'], block['h']; txt_center_x = txt_x + txt_w / 2
        if txt_y >= img_bottom - 10: 
             vertical_dist = txt_y - img_bottom; overlap_x = max(0, min(img_x + img_w, txt_x + txt_w) - max(img_x, txt_x)); center_dist_x = abs(txt_center_x - img_center_x)
             if -10 <= vertical_dist < max_distance_y and (overlap_x > img_w * 0.2 or center_dist_x < img_w * 0.5):
                 dist = max(0, vertical_dist) + center_dist_x * 0.2 
                 if dist < min_dist: min_dist = dist; best_caption = block['text']; best_block_idx = i
        elif txt_x >= img_right - 10 and best_block_idx == -1: 
             horizontal_dist = txt_x - img_right; overlap_y = max(0, min(img_y + img_h, txt_y + txt_h) - max(img_y, txt_y)); center_dist_y = abs((txt_y + txt_h/2) - (img_y + img_h/2))
             if -10 <= horizontal_dist < max_distance_x and (overlap_y > img_h * 0.2 or center_dist_y < img_h * 0.5):
                  dist = max(0, horizontal_dist) + center_dist_y * 0.2
                  if dist < min_dist: min_dist = dist; best_caption = block['text']; best_block_idx = i
    if best_block_idx != -1 and best_block_idx < len(text_blocks): 
        if text_blocks[best_block_idx].get('type') != "Header": text_blocks[best_block_idx]['used_as_caption'] = True
    return best_caption

# --- Main Processing Function ---
def extract_images_and_text(image_path, output_dir, config):
    """Extracts images and text, associating text with images."""
    results = []; base_filename = os.path.splitext(os.path.basename(image_path))[0]
    try:
        img = cv2.imread(image_path)
        if img is None: print(f"Error: Could not read image {image_path}"); return results
        img_height, img_width = img.shape[:2]; total_image_area = img_height * img_width
        print(f"\nProcessing {image_path} ({img_width}x{img_height})")

        print("  Performing OCR...")
        try:
            ocr_data = pytesseract.image_to_data(img, output_type=Output.DATAFRAME, lang='eng') 
            # Removed previous debug prints here

            if not isinstance(ocr_data, pd.DataFrame) or ocr_data.empty:
                 print(f"Warning: OCR for {image_path} returned empty or non-DataFrame result.")
                 ocr_data = pd.DataFrame() 
            # Removed further processing here, it's handled in get_text_blocks now
            print(f"  OCR initial result rows: {len(ocr_data)}")
        except pytesseract.TesseractError as ocr_err: print(f"  Tesseract Error during OCR for {image_path}: {ocr_err}"); ocr_data = pd.DataFrame()
        except Exception as e: print(f"  Unexpected Error during OCR setup/execution for {image_path}: {e}"); ocr_data = pd.DataFrame()

        text_blocks = get_text_blocks(ocr_data) # Pass raw (but checked type) DF
        print(f"  Grouped into {len(text_blocks)} text lines/blocks (conf > 10).")
        
        header_text = []; footer_text = []; location_text = []; additional_info_text = []
        for i, block in enumerate(text_blocks):
            block_type = classify_text_block(block, i, text_blocks, img_width, img_height)
            block['type'] = block_type 
            if block_type == "Header": header_text.append(block.get('text','')) 
            elif block_type == "Footer": footer_text.append(block.get('text',''))
            elif block_type == "Location": location_text.append(block.get('text',''))
            elif block_type == "Additional Info": additional_info_text.append(block.get('text',''))
            if block_type in ["Header", "Footer", "Location", "Additional Info"]: block['used_as_caption'] = True
            else: block['used_as_caption'] = False

        page_metadata = {
            "Header": " | ".join(filter(None, header_text)), "Footer": " | ".join(filter(None, footer_text)),
            "Location": " ".join(filter(None, location_text)), "Additional Info": " | ".join(filter(None, additional_info_text))
        }
        print(f"  Page Metadata: {page_metadata}") # Check this line for the header

        # --- Image Detection and Captioning ---
        print("  Detecting image regions...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); blurred = cv2.GaussianBlur(gray, (5, 5), 0)
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
                if (config['min_area_ratio'] < area_ratio < config['max_area_ratio'] and
                    config['aspect_ratio_min'] < aspect_ratio < config['aspect_ratio_max']):
                    valid_bounding_boxes.append((x, y, w, h))
        valid_bounding_boxes.sort(key=lambda box: (box[1], box[0]))
        print(f"  Found {len(valid_bounding_boxes)} potential image regions.")

        os.makedirs(output_dir, exist_ok=True); extracted_count = 0
        for i, (x, y, w, h) in enumerate(valid_bounding_boxes):
            image_bbox = (x, y, w, h)
            caption = find_caption_for_image(image_bbox, text_blocks, 
                                             max_distance_y=config['caption_max_dist_y'], max_distance_x=config['caption_max_dist_x'])
            padding = config['padding']; y1 = max(0, y - padding); y2 = min(img_height, y + h + padding)
            x1 = max(0, x - padding); x2 = min(img_width, x + w + padding)
            if y1 >= y2 or x1 >= x2: print(f"    Warning: Invalid crop dimensions for image {i+1}. Skipping save."); continue
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img is None or cropped_img.size == 0: print(f"    Warning: Cropped image {i+1} is empty. Skipping save."); continue
            output_filename = os.path.join(output_dir, f"{base_filename}_img_{i+1:02d}.jpg")
            try:
                success = cv2.imwrite(output_filename, cropped_img)
                if not success: print(f"    Error: OpenCV failed to save image {output_filename}."); continue 
                results.append({
                    "source_page": image_path, "extracted_image": output_filename, "image_bounding_box": [int(x), int(y), int(w), int(h)], 
                    "page_header": page_metadata["Header"], "page_footer": page_metadata["Footer"],
                    "page_location": page_metadata["Location"], "page_additional_info": page_metadata["Additional Info"],
                    "image_caption": caption })
                extracted_count += 1
            except Exception as e: print(f"    Error during image save or metadata append for {output_filename}: {e}")
        print(f"  Successfully processed and potentially saved {extracted_count} images from {image_path}")
        return results
    except Exception as e: print(f"An unexpected major error occurred processing {image_path}: {e}"); traceback.print_exc(); return [] 

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract individual images and associated text from scanned pages.")
    # ... (Argument parsing remains the same) ...
    parser.add_argument("input_folder", help="Folder containing the scanned image files.")
    parser.add_argument("output_folder", help="Folder where extracted images and metadata JSON will be saved.")
    parser.add_argument("--min_area", type=float, default=0.005) 
    parser.add_argument("--max_area", type=float, default=0.4)   
    parser.add_argument("--min_aspect", type=float, default=0.3) 
    parser.add_argument("--max_aspect", type=float, default=3.0) 
    parser.add_argument("--padding", type=int, default=5)       
    parser.add_argument("--threshold", type=int, default=235)    
    parser.add_argument("--caption_max_dist_y", type=int, default=75) 
    parser.add_argument("--caption_max_dist_x", type=int, default=100)
    parser.add_argument('--ext', nargs='+', default=['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']) 
    parser.add_argument('--output_json', default="metadata.json")
    parser.add_argument('--tesseract_cmd', default=None)
    args = parser.parse_args()

    # --- Tesseract Check ---
    # ... (remains the same) ...
    if args.tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    try: version_info = pytesseract.get_tesseract_version(); print(f"Using Tesseract OCR version: {version_info}")
    except pytesseract.TesseractNotFoundError: print(f"\nError: Tesseract executable not found.\nAttempted path: {pytesseract.pytesseract.tesseract_cmd}\nPlease install Tesseract OCR and ensure it's in your system's PATH,\nor provide the correct path using the --tesseract_cmd argument."); exit(1) 
    except Exception as e: print(f"\nError accessing Tesseract: {e}\nThere might be an issue with your Tesseract installation or the provided path."); exit(1)

    valid_extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in args.ext]
    all_metadata = []; processed_files = 0
    config = { 'min_area_ratio': args.min_area, 'max_area_ratio': args.max_area,'aspect_ratio_min': args.min_aspect, 'aspect_ratio_max': args.max_aspect,'padding': args.padding, 'threshold_value': args.threshold,'caption_max_dist_y': args.caption_max_dist_y, 'caption_max_dist_x': args.caption_max_dist_x }
    print(f"\nScanning folder: {args.input_folder}") # ... (print other settings) ...
    print(f"Outputting images to: {args.output_folder}")
    print(f"Outputting metadata to: {os.path.join(args.output_folder, args.output_json)}")
    print(f"Processing extensions: {valid_extensions}")
    print("-" * 30)
    os.makedirs(args.output_folder, exist_ok=True)

    # --- Main Loop ---
    # ... (remains the same) ...
    for filename in sorted(os.listdir(args.input_folder)): 
        input_path = os.path.join(args.input_folder, filename)
        if os.path.isfile(input_path) and os.path.splitext(filename)[1].lower() in valid_extensions:
            processed_files += 1
            page_results = extract_images_and_text(input_path, args.output_folder, config)
            if page_results: all_metadata.extend(page_results)
            print("-" * 30)
        else: pass

    # --- Save Metadata ---
    # ... (remains the same) ...
    output_json_path = os.path.join(args.output_folder, args.output_json)
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f: json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        print(f"\nMetadata saved to {output_json_path}")
    except TypeError as te: print(f"\nError saving metadata JSON: Data might not be JSON serializable. Error: {te}")
    except Exception as e: print(f"\nError saving metadata JSON: {e}")
    print(f"\n18-Apr: Batch processing complete. Processed {processed_files} files. Extracted {len(all_metadata)} images with metadata.")