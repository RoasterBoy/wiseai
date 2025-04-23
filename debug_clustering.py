import os
import sys
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from PIL import Image
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import difflib
import re
import pandas as pd
import argparse
import time
from datetime import timedelta
import traceback

def check_dependencies():
    """Check if all required dependencies are installed and working."""
    print("Checking dependencies...")
    
    # Check pytesseract
    try:
        pytesseract.get_tesseract_version()
        print("✓ Tesseract OCR is installed")
    except Exception as e:
        print(f"✗ Tesseract OCR issue: {e}")
        print("Make sure Tesseract OCR is installed on your system")
        print("Installation instructions: https://github.com/tesseract-ocr/tesseract")
        return False
    
    # Check pdf2image and poppler
    try:
        # Just import to check
        from pdf2image import convert_from_path
        print("✓ pdf2image is installed")
        # We'll check poppler during actual conversion
    except Exception as e:
        print(f"✗ pdf2image issue: {e}")
        return False
    
    # Check OpenCV
    try:
        cv2.__version__
        print("✓ OpenCV is installed")
    except Exception as e:
        print(f"✗ OpenCV issue: {e}")
        return False
    
    # Check scikit-learn
    try:
        from sklearn import __version__
        print("✓ scikit-learn is installed")
    except Exception as e:
        print(f"✗ scikit-learn issue: {e}")
        return False
    
    return True

def process_pdf(pdf_path, output_dir, pdf_index=0):
    """Process a single PDF file and extract text."""
    print(f"\nProcessing PDF: {pdf_path}")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, "enhanced_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "text_output"), exist_ok=True)
    
    try:
        # Convert PDF to images
        print("Converting PDF to images...")
        pages = convert_from_path(pdf_path, dpi=300)
        print(f"Successfully converted PDF to {len(pages)} images")
        
        for i, page in enumerate(pages[:1]):  # Process only the first page for testing
            print(f"Processing page {i+1} of {len(pages)}")
            
            # Create metadata
            metadata = {
                'pdf_file': os.path.basename(pdf_path),
                'pdf_index': pdf_index,
                'page_number': i + 1,
            }
            
            # Preprocess the image
            print("Preprocessing image...")
            gray_image = page.convert('L')
            opencv_image = np.array(gray_image)
            opencv_image = cv2.convertScaleAbs(opencv_image, alpha=1.5, beta=0)
            blurred = cv2.GaussianBlur(opencv_image, (3, 3), 0)
            threshold_image = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            enhanced_image = Image.fromarray(threshold_image)
            
            # Save enhanced image
            print("Saving enhanced image...")
            image_filename = f"{pdf_index}_{i+1}.png"
            enhanced_image_path = os.path.join(output_dir, "enhanced_images", image_filename)
            enhanced_image.save(enhanced_image_path)
            print(f"Enhanced image saved to: {enhanced_image_path}")
            
            # Extract text using Tesseract OCR
            print("Extracting text using OCR...")
            try:
                custom_config = r'--oem 1 --psm 3 -l eng'
                text = pytesseract.image_to_string(enhanced_image, config=custom_config)
                print(f"Successfully extracted {len(text)} characters of text")
                
                # Save extracted text
                text_filename = f"{pdf_index}_{i+1}.txt"
                text_path = os.path.join(output_dir, "text_output", text_filename)
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
                print(f"Text saved to: {text_path}")
                
                # Print a snippet of the extracted text
                print("\nExtracted text snippet:")
                print(text[:200] + "..." if len(text) > 200 else text)
                
            except Exception as e:
                print(f"Error during OCR: {e}")
                traceback.print_exc()
        
        print(f"Successfully processed PDF: {pdf_path}")
        return True
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Debug Document Clustering')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing PDF files or a single PDF file')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Starting Document Clustering Debug Script")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Check if input is a file or directory
    if os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
        pdf_files = [args.input]
        input_dir = os.path.dirname(args.input)
    elif os.path.isdir(args.input):
        input_dir = args.input
        pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                     if f.lower().endswith('.pdf')]
    else:
        print(f"Error: Input path {args.input} is not a valid PDF file or directory")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Check dependencies
    if not check_dependencies():
        print("Error: Missing or misconfigured dependencies")
        return
    
    # Process PDFs
    print(f"Found {len(pdf_files)} PDF files to process")
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
        
    # Process only the first PDF file for testing
    if pdf_files:
        start_time = time.time()
        success = process_pdf(pdf_files[0], args.output)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nProcessing took {duration:.2f} seconds")
        
        if success:
            print("Debug test completed successfully")
        else:
            print("Debug test failed")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()
