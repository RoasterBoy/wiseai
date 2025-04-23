"""
Text Extractor module for Document Clustering System.

This module handles image preprocessing, OCR, and language detection.
"""

import os
import logging
import re
import traceback
import numpy as np
import cv2
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

class TextExtractor:
    """
    Handles the extraction of text from images, including preprocessing,
    OCR, language detection, and document type classification.
    """
    
    def __init__(self, output_dir, save_enhanced_images=False, save_text_output=True):
        """
        Initialize the text extractor.
        
        Args:
            output_dir: Directory to save enhanced images and text files
            save_enhanced_images: Whether to save preprocessed images
            save_text_output: Whether to save extracted text files
        """
        self.output_dir = output_dir
        self.save_enhanced_images = save_enhanced_images
        self.save_text_output = save_text_output
    
    def preprocess_image(self, image):
        """
        Optimize image for OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image object
        """
        try:
            # Convert to NumPy array for OpenCV processing
            opencv_image = np.array(image)
            
            # If image is RGB, convert to grayscale
            if len(opencv_image.shape) == 3:
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = opencv_image
            
            # Apply bilateral filter for noise reduction while preserving edges
            denoised = cv2.bilateralFilter(gray, 5, 75, 75)
            
            # Use adaptive thresholding
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # Block size
                2    # Constant subtracted from mean
            )
            
            # Apply morphological operations
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL for Tesseract compatibility
            return Image.fromarray(cleaned)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            logger.debug(traceback.format_exc())
            # Return original image if processing fails
            return image
    
    def detect_language(self, text):
        """
        Detect language from text using common word frequency.
        
        Args:
            text: Extracted text
            
        Returns:
            Detected language code ('eng', 'spa', 'fra')
        """
        if not text or len(text.strip()) < 20:
            return 'eng'  # Default to English for very short or empty text
            
        text_lower = text.lower()
        
        # Common words by language
        eng_words = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but']
        spa_words = ['el', 'la', 'que', 'de', 'en', 'y', 'un', 'ser', 'por', 'con']
        fra_words = ['le', 'la', 'les', 'un', 'une', 'et', 'de', 'que', 'pas', 'pour']
        
        # Count occurrences (with word boundaries)
        eng_count = sum(1 for word in eng_words if f" {word} " in text_lower 
                      or text_lower.startswith(f"{word} ") or text_lower.endswith(f" {word}"))
        spa_count = sum(1 for word in spa_words if f" {word} " in text_lower 
                      or text_lower.startswith(f"{word} ") or text_lower.endswith(f" {word}"))
        fra_count = sum(1 for word in fra_words if f" {word} " in text_lower 
                      or text_lower.startswith(f"{word} ") or text_lower.endswith(f" {word}"))
        
        # Normalize by the number of words in each language list
        eng_score = eng_count / len(eng_words)
        spa_score = spa_count / len(spa_words)
        fra_score = fra_count / len(fra_words)
        
        # Get the language with the highest score
        scores = {'eng': eng_score, 'spa': spa_score, 'fra': fra_score}
        max_lang = max(scores.items(), key=lambda x: x[1])
        
        # Return the detected language, or default to English if no clear winner
        return max_lang[0] if max_lang[1] > 0.1 else 'eng'  # Threshold to avoid random matches
    
    def detect_document_type(self, text, language='eng'):
        """
        Detect document type based on text patterns, with multilingual support.
        
        Args:
            text: Extracted text
            language: Detected language code ('eng', 'spa', 'fra')
            
        Returns:
            Document type string ('receipt', 'letter', 'report', 'form', or 'document')
        """
        if not text or len(text.strip()) < 20:
            return 'unknown'
            
        text_lower = text.lower()
        
        # Language-specific pattern dictionaries
        patterns = {
            'eng': {
                'receipt': r'receipt|total|subtotal|payment|paid|amount|\$|€|£',
                'letter': r'dear\s+[a-z,\.]+|sincerely|regards|yours truly',
                'report': r'report|analysis|summary|findings|conclusion|introduction',
                'form': r'form|please fill|checkbox|check one|signature|date of birth'
            },
            'spa': {
                'receipt': r'recibo|total|subtotal|pago|pagado|importe|\$|€|£',
                'letter': r'estimado|querido|atentamente|saludos|cordialmente',
                'report': r'informe|análisis|resumen|hallazgos|conclusión|introducción',
                'form': r'formulario|por favor|casilla|marque|firma|fecha de nacimiento'
            },
            'fra': {
                'receipt': r'reçu|total|sous-total|paiement|payé|montant|\$|€|£',
                'letter': r'cher\s+[a-z,\.]+|cordialement|salutations|sincèrement',
                'report': r'rapport|analyse|résumé|résultats|conclusion|introduction',
                'form': r'formulaire|s\'il vous plaît|case à cocher|signature|date de naissance'
            }
        }
        
        # Use English patterns as fallback if language not supported
        lang_patterns = patterns.get(language, patterns['eng'])
        
        # Check for receipt patterns
        if re.search(lang_patterns['receipt'], text_lower) and \
           re.search(r'\d+[\.,]\d{2}', text) and len(text) < 1000:
            return 'receipt'
        
        # Check for letter patterns
        if re.search(lang_patterns['letter'], text_lower):
            return 'letter'
        
        # Check for report patterns
        if re.search(lang_patterns['report'], text_lower) and len(text) > 1000:
            return 'report'
            
        # Check for form patterns
        if re.search(lang_patterns['form'], text_lower):
            return 'form'
            
        # Default
        return 'document'
    
    def extract_text_from_image(self, image, metadata):
        """
        Extract text from an image using OCR, with optimized language handling.
        
        Args:
            image: PIL Image object
            metadata: Dictionary with page metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        # Preprocess the image
        enhanced_image = self.preprocess_image(image)
        
        # Save enhanced image only if enabled
        if self.save_enhanced_images:
            try:
                image_filename = f"{metadata['pdf_index']}_{metadata['page_number']}.png"
                enhanced_image_path = os.path.join(self.output_dir, "enhanced_images", image_filename)
                enhanced_image.save(enhanced_image_path)
                metadata['enhanced_image_path'] = enhanced_image_path
                logger.debug(f"Saved enhanced image: {enhanced_image_path}")
            except Exception as e:
                logger.error(f"Error saving enhanced image: {e}")
                metadata['enhanced_image_path'] = None
        else:
            metadata['enhanced_image_path'] = None
        
        # Extract text using Tesseract OCR
        try:
            # Use multi-language mode directly
            custom_config = r'--oem 1 --psm 3 -l eng+spa+fra'
            text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Detect language from the extracted text
            detected_lang = self.detect_language(text)
            
            # Save extracted text if enabled
            if self.save_text_output:
                try:
                    text_filename = f"{metadata['pdf_index']}_{metadata['page_number']}.txt"
                    text_path = os.path.join(self.output_dir, "text_output", text_filename)
                    with open(text_path, 'w', encoding='utf-8') as text_file:
                        text_file.write(text)
                    metadata['text_path'] = text_path
                    logger.debug(f"Saved text to: {text_path}")
                except Exception as e:
                    logger.error(f"Error saving text file: {e}")
                    metadata['text_path'] = None
            else:
                metadata['text_path'] = None
            
            # Update metadata
            metadata['text_length'] = len(text)
            metadata['has_text'] = len(text.strip()) > 0
            metadata['language'] = detected_lang
            
            # Detect document type based on text patterns
            metadata['doc_type'] = self.detect_document_type(text, detected_lang)
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            logger.debug(traceback.format_exc())
            metadata['text_path'] = None
            metadata['text_length'] = 0
            metadata['has_text'] = False
            metadata['language'] = 'unknown'
            metadata['doc_type'] = 'unknown'
            return "", metadata