#!/usr/bin/env python3
"""
Simple Document Processor - A simplified version of the document clustering prototype
with verbose logging and robust error handling
"""

import os
import sys
import logging
import argparse
import time
from datetime import timedelta
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("doc_processor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_dependencies():
    """Import dependencies with detailed error reporting"""
    try:
        global pytesseract, Image, convert_from_path, np, cv2, pd
        
        logger.info("Importing pytesseract...")
        import pytesseract
        logger.info(f"pytesseract imported successfully")
        
        logger.info("Importing PIL.Image...")
        from PIL import Image
        logger.info(f"PIL.Image imported successfully")
        
        logger.info("Importing pdf2image...")
        from pdf2image import convert_from_path
        logger.info(f"pdf2image imported successfully")
        
        logger.info("Importing numpy...")
        import numpy as np
        logger.info(f"numpy imported successfully")
        
        logger.info("Importing OpenCV...")
        import cv2
        logger.info(f"OpenCV imported successfully, version: {cv2.__version__}")
        
        logger.info("Importing pandas...")
        import pandas as pd
        logger.info(f"pandas imported successfully")
        
        # Check Tesseract path
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {tesseract_version}")
        except Exception as e:
            logger.error(f"Error getting Tesseract version: {e}")
            if sys.platform.startswith('win'):
                logger.info("On Windows, you may need to set tesseract_cmd explicitly:")
                logger.info("pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
            raise
            
        return True
    except ImportError as e:
        logger.error(f"Failed to import required dependency: {e}")
        logger.error("Please install missing dependencies:")
        logger.error("pip install pytesseract pdf2image pillow numpy opencv-python pandas")
        logger.error("Also ensure Tesseract OCR and Poppler are installed on your system")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        logger.error(traceback.format_exc())
        return False

def process_pdf(input_path, output_dir, max_pages=None):
    """
    Process a single PDF file
    
    Args:
        input_path: Path to the PDF file
        output_dir: Directory to save results
        max_pages: Maximum number of pages to process (None for all)
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return False
        
    # Create output subdirectories
    enhanced_dir = os.path.join(output_dir, "enhanced_images")
    text_dir = os.path.join(output_dir, "text_output")
    
    for directory in [enhanced_dir, text_dir]:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)
    
    # Process the PDF
    logger.info(f"Processing PDF: {input_path}")
    
    try:
        start_time = time.time()
        
        # Converting PDF to images
        logger.info("Converting PDF to images...")
        
        # On Windows, you might need to specify the poppler_path
        if sys.platform.startswith('win'):
            # Check if poppler_path environment variable is set
            poppler_path = os.environ.get('POPPLER_PATH')
            if poppler_path:
                logger.info(f"Using Poppler path from environment: {poppler_path}")
                images = convert_from_path(input_path, dpi=300, poppler_path=poppler_path)
            else:
                logger.info("Attempting to convert PDF without specifying poppler_path")
                logger.info("If this fails, set the POPPLER_PATH environment variable")
                images = convert_from_path(input_path, dpi=300)
        else:
            images = convert_from_path(input_path, dpi=300)
        
        logger.info(f"Converted PDF to {len(images)} images")
        
        if max_pages is not None and max_pages > 0:
            logger.info(f"Limiting processing to {max_pages} pages")
            images = images[:max_pages]
        
        # Process each page
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1} of {len(images)}")
            
            # Preprocess the image for better OCR results
            logger.info("Preprocessing image...")
            try:
                # Convert to grayscale
                gray_image = image.convert('L')
                
                # Convert to numpy array for OpenCV processing
                np_image = np.array(gray_image)
                
                # Apply contrast enhancement
                enhanced_image = cv2.convertScaleAbs(np_image, alpha=1.5, beta=0)
                
                # Apply Gaussian blur to reduce noise
                blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
                
                # Apply adaptive thresholding
                threshold_image = cv2.adaptiveThreshold(
                    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # Convert back to PIL Image
                pil_enhanced = Image.fromarray(threshold_image)
                
                # Save enhanced image
                image_filename = f"page_{i+1}.png"
                image_path = os.path.join(enhanced_dir, image_filename)
                logger.info(f"Saving enhanced image to {image_path}")
                pil_enhanced.save(image_path)
                
            except Exception as e:
                logger.error(f"Error preprocessing image: {e}")
                logger.error(traceback.format_exc())
                continue
            
            # Perform OCR
            logger.info("Performing OCR...")
            try:
                # Use Tesseract to extract text
                text = pytesseract.image_to_string(pil_enhanced)
                
                # Save the extracted text
                text_filename = f"page_{i+1}.txt"
                text_path = os.path.join(text_dir, text_filename)
                logger.info(f"Saving extracted text to {text_path}")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Log a snippet of the extracted text
                snippet = text[:100] + '...' if len(text) > 100 else text
                logger.info(f"Text snippet: {snippet}")
                
            except Exception as e:
                logger.error(f"Error performing OCR: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Calculate processing time
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"PDF processing completed in {timedelta(seconds=duration)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple Document Processor')
    parser.add_argument('--input', '-i', required=True, help='Input PDF file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--pages', '-p', type=int, help='Maximum pages to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Starting Simple Document Processor")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max pages: {args.pages if args.pages else 'All'}")
    
    # Setup dependencies
    if not setup_dependencies():
        logger.error("Failed to setup dependencies")
        return 1
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        logger.info(f"Creating output directory: {args.output}")
        os.makedirs(args.output)
    
    # Process PDF
    if not process_pdf(args.input, args.output, args.pages):
        logger.error("PDF processing failed")
        return 1
    
    logger.info("PDF processing completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
