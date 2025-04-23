# filename: optimized_document_clustering_fixed_v2.py
import os
import sys
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import difflib
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse
import time
from datetime import timedelta
import pickle
# Use try-except for langdetect import as it might not be critical for core functionality
try:
    from langdetect import detect
except ImportError:
    print("Warning: 'langdetect' library not found. Language detection features will be limited.")
    detect = None # Define detect as None if library is missing

import shutil
import warnings
from tqdm import tqdm  # For progress bars
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

class OptimizedDocumentClustering:
    def __init__(self, input_dir, output_dir, sample_size=None, max_workers=None, use_cache=True):
        """
        Initialize the optimized document clustering.

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save results
            sample_size: Number of PDFs to process (None for all)
            max_workers: Maximum number of parallel workers (None for CPU count)
            use_cache: Whether to use caching
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.use_cache = use_cache
        self.pdf_files = []
        self.page_images = [] # Consider removing if full images aren't needed long-term
        self.page_texts = []
        self.page_metadata = []
        self.document_clusters = []

        # Timing information
        self.start_time = None
        self.file_times = {}
        self.total_time = 0

        # Create output directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.output_dir,
            os.path.join(self.output_dir, "enhanced_images"),
            os.path.join(self.output_dir, "text_output"),
            os.path.join(self.output_dir, "document_clusters"), # For organized output links
            os.path.join(self.output_dir, "cache")
        ]

        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    def load_pdf_files(self):
        """Load PDF files from the input directory."""
        try:
            all_files = os.listdir(self.input_dir)
            self.pdf_files = [f for f in all_files if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(self.input_dir, f))]
        except FileNotFoundError:
            print(f"Error: Input directory not found: {self.input_dir}")
            self.pdf_files = []
            return # Exit early if input dir doesn't exist

        if not self.pdf_files:
             print(f"Warning: No PDF files found in {self.input_dir}")
             return

        if self.sample_size:
            self.pdf_files = self.pdf_files[:min(self.sample_size, len(self.pdf_files))]

        print(f"Found {len(self.pdf_files)} PDF files to process")

    def check_cache(self, pdf_file):
        """Check if results for this PDF are already cached."""
        if not self.use_cache:
            return None

        cache_path = os.path.join(self.output_dir, "cache", f"{pdf_file}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    # Load only metadata and text from cache to save memory
                    cached_data = pickle.load(f)
                    if isinstance(cached_data, tuple) and len(cached_data) == 2:
                         # Expected format: (metadata_list, texts)
                         return cached_data
                    else:
                         # Older format might include images, try to load selectively
                         # This part depends on the exact structure of the old cache format
                         # Assuming older format was (images, metadata, texts)
                         if isinstance(cached_data, tuple) and len(cached_data) == 3:
                              print(f"Note: Loading from older cache format for {pdf_file}. Images ignored.")
                              _, metadata_list, texts = cached_data
                              return metadata_list, texts
                         else:
                              print(f"Warning: Unexpected cache format for {pdf_file}. Ignoring cache.")
                              return None

            except Exception as e:
                print(f"Error loading cache for {pdf_file}: {e}. Ignoring cache.")
                # Optionally remove corrupted cache file: os.remove(cache_path)
        return None

    def save_to_cache(self, pdf_file, metadata_list, texts):
        """Save results (metadata and text only) to cache."""
        if not self.use_cache:
            return

        cache_dir = os.path.join(self.output_dir, "cache")
        # No need to makedirs here, done in __init__/_create_directories
        # os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{pdf_file}.pkl")
        try:
            # Save only metadata and texts
            results_to_cache = (metadata_list, texts)
            with open(cache_path, 'wb') as f:
                pickle.dump(results_to_cache, f)
        except Exception as e:
            print(f"Error saving cache for {pdf_file}: {e}")

    def preprocess_image(self, image):
        """
        Optimize image quality for OCR with faster processing.

        Args:
            image: PIL Image object

        Returns:
            Enhanced PIL Image object ready for OCR
        """
        try:
            # Convert to grayscale
            gray_image = image.convert('L')

            # Use OpenCV for faster processing
            opencv_image = np.array(gray_image)

            # Apply basic contrast enhancement (Adjust alpha/beta as needed)
            # opencv_image = cv2.convertScaleAbs(opencv_image, alpha=1.5, beta=0)

            # Apply mild Gaussian blur to reduce noise (faster than median filter)
            # Kernel size (e.g., (3, 3)) can be tuned
            blurred = cv2.GaussianBlur(opencv_image, (3, 3), 0)

            # Apply adaptive thresholding (better than simple thresholding for varying lighting)
            # Block size and C constant can be tuned (e.g., 11, 2)
            threshold_image = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Convert back to PIL Image
            enhanced_image = Image.fromarray(threshold_image)

            return enhanced_image
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            # Return original image or grayscale if preprocessing fails
            return image.convert('L')


    def extract_text_from_image(self, image, metadata):
        """
        Extract text from an image using OCR, with optimized language detection
        and potentially improved handwriting recognition settings.

        Args:
            image: PIL Image object (original, preprocessing happens inside)
            metadata: Dictionary with page metadata (will be updated)

        Returns:
            Tuple of (extracted text, updated metadata)
        """
        enhanced_image = None # Initialize
        try:
            # Preprocess the image
            enhanced_image = self.preprocess_image(image)

            # Save enhanced image (optional, can consume disk space)
            image_filename = f"enhanced_{metadata['pdf_index']}_{metadata['page_number']}.png"
            enhanced_image_path = os.path.join(self.output_dir, "enhanced_images", image_filename)
            enhanced_image.save(enhanced_image_path)

            # Update metadata with path
            metadata['enhanced_image_path'] = enhanced_image_path

        except Exception as e:
             print(f"Error saving enhanced image for page {metadata.get('page_number')} of {metadata.get('pdf_file')}: {e}")
             metadata['enhanced_image_path'] = None
             # If saving failed, try OCR on the original image (or grayscale)
             if enhanced_image is None:
                  enhanced_image = image.convert('L')


        # --- Tesseract OCR Execution ---
        text = ""
        try:
            # --- MODIFIED TESSERACT CONFIGURATION ---
            # Use Tesseract's language detection capabilities
            # Changed --psm from 3 to 6 (Assume a single uniform block of text)
            # Added script/Latin for potentially better handwriting recognition
            # Ensure 'script/Latin' traineddata is installed for Tesseract
            # Common PSM modes to try:
            # 3: Fully automatic page segmentation, but no OSD. (Default)
            # 4: Assume a single column of text of variable sizes.
            # 6: Assume a single uniform block of text. (Often good for invoices/receipts/handwriting)
            # 11: Sparse text. Find as much text as possible in no particular order.
            # 12: Sparse text with OSD.
            custom_config = r'--oem 1 --psm 6 -l eng+spa+fra+script/Latin'
            # --- END MODIFICATION ---

            # Perform OCR
            text = pytesseract.image_to_string(enhanced_image, config=custom_config)

        except pytesseract.TesseractNotFoundError:
             print("Error: Tesseract is not installed or not in your PATH.")
             # Exit or handle appropriately if Tesseract is required
             sys.exit("Tesseract not found. Please install Tesseract.")
        except Exception as e:
            print(f"Error during Tesseract OCR for page {metadata.get('page_number')} of {metadata.get('pdf_file')}: {e}")
            # Text remains ""

        # --- Post-OCR Processing ---
        # Default language
        detected_lang = 'eng' # Default to English if langdetect fails or is disabled

        # Attempt language detection if library is available and text is sufficient
        if detect and text and len(text.strip()) > 30: # Use a reasonable threshold
            try:
                lang_code = detect(text)
                # Map langdetect codes to tesseract codes (or keep langdetect codes)
                lang_map = {'en': 'eng', 'es': 'spa', 'fr': 'fra'} # Add more mappings if needed
                detected_lang = lang_map.get(lang_code, lang_code) # Fallback to detected code if no mapping
            except Exception as lang_e:
                 # Keep default language if detection fails
                 # print(f"Language detection failed for page {metadata.get('page_number')}: {lang_e}")
                 pass

        # Save extracted text to file
        text_path = None
        try:
            text_filename = f"text_{metadata['pdf_index']}_{metadata['page_number']}.txt"
            text_path = os.path.join(self.output_dir, "text_output", text_filename)
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
        except Exception as e:
             print(f"Error saving text file {text_path}: {e}")
             text_path = None # Ensure path is None if saving failed


        # Update metadata dictionary
        metadata['text_path'] = text_path
        metadata['text_length'] = len(text)
        metadata['has_text'] = len(text.strip()) > 0
        metadata['language'] = detected_lang

        # Detect document type based on text patterns
        metadata['doc_type'] = self.detect_document_type(text, detected_lang)

        return text, metadata


    def detect_document_type(self, text, language='eng'):
        """
        Detect document type based on text patterns, with multilingual support.
        Simple example, can be expanded significantly.

        Args:
            text: Extracted text
            language: Detected language code (e.g., 'eng', 'spa', 'fra', 'de')

        Returns:
            Document type string (e.g., 'receipt', 'letter', 'report', 'form', 'document')
        """
        if not text: # No text, can't determine type from content
            return 'unknown'

        text_lower = text.lower()

        # Language-specific pattern dictionaries (expand as needed)
        # Using raw strings (r'...') for all regex patterns
        # Corrected the ':' vs '\:' issue
        patterns = {
            'eng': {
                'receipt': r'receipt|invoice|total due|subtotal|payment|paid|amount due|\$|€|£|vat|tax invoice',
                'letter': r'dear\s+[a-z,\.]+|sincerely|regards|yours truly| attn:| re:', # Corrected ':' here
                'report': r'report|analysis|summary|findings|conclusion|introduction|executive summary|appendix|figure \d+|table \d+',
                'form': r'form|please fill|checkbox|check one|signature:|date:|agreement|terms and conditions|social security number|date of birth'
            },
            'spa': {
                'receipt': r'recibo|factura|total|subtotal|pago|pagado|importe|\$|€|£|iva',
                'letter': r'estimado|querido|atentamente|saludos|cordialmente',
                'report': r'informe|análisis|resumen|hallazgos|conclusión|introducción|anexo|figura \d+|tabla \d+',
                'form': r'formulario|por favor|casilla|marque|firma:|fecha:|acuerdo|términos y condiciones|número de seguro social|fecha de nacimiento'
            },
            'fra': {
                'receipt': r'reçu|facture|total|sous-total|paiement|payé|montant|\$|€|£|tva',
                'letter': r'cher\s+[a-z,\.]+|cordialement|salutations|sincèrement',
                'report': r'rapport|analyse|résumé|résultats|conclusion|introduction|annexe|figure \d+|tableau \d+',
                'form': r'formulaire|s\'il vous plaît|case à cocher|signature:|date:|accord|termes et conditions|numéro de sécurité sociale|date de naissance'
            },
            'de': {
                'receipt': r'quittung|rechnung|gesamtbetrag|zwischensumme|zahlung|bezahlt|betrag|\$|€|£|mwst|ust',
                'letter': r'sehr geehrte|mit freundlichen grüßen|hochachtungsvoll',
                'report': r'bericht|analyse|zusammenfassung|ergebnisse|schlussfolgerung|einführung|anhang|abbildung \d+|tabelle \d+',
                'form': r'formular|bitte füllen|kontrollkästchen|unterschrift:|datum:|vereinbarung|agb|sozialversicherungsnummer|geburtsdatum'
            }
        }

        # Use specific language patterns if available, otherwise default to English
        lang_key = language if language in patterns else 'eng'
        if lang_key not in patterns:
            lang_key = 'eng'
        lang_patterns = patterns[lang_key]


        # Check patterns in a specific order (e.g., more specific first)
        # Receipt check: Keywords + currency + typically shorter text
        if re.search(lang_patterns['receipt'], text_lower) and \
           (re.search(r'\d+[\.,]\d{2}\b', text) or re.search(r'[\$€£]', text)) and \
           len(text) < 2000: # Heuristic length limit for receipts/invoices
            return 'receipt'

        # Form check: Keywords
        if re.search(lang_patterns['form'], text_lower):
            return 'form'

        # Letter check: Keywords
        if re.search(lang_patterns['letter'], text_lower):
            return 'letter'

        # Report check: Keywords + typically longer text
        if re.search(lang_patterns['report'], text_lower) and len(text) > 1500: # Heuristic length
            return 'report'

        # Default type if no specific pattern matches
        return 'document'


    def process_single_pdf(self, pdf_path, pdf_index, pdf_filename):
        """
        Process a single PDF file: Convert pages to images and extract text.

        Args:
            pdf_path: Path to the PDF file
            pdf_index: Index of the PDF file (for unique naming)
            pdf_filename: Name of the PDF file (for caching key)

        Returns:
            Tuple of (list of page metadata dicts, list of page texts)
            Returns (None, None) if processing fails critically.
        """
        # Check cache first
        cached_results = self.check_cache(pdf_filename)
        if cached_results:
            metadata_list, texts = cached_results
            # Ensure essential keys are present in cached metadata
            for i, meta in enumerate(metadata_list):
                 meta.setdefault('pdf_file', pdf_filename)
                 meta.setdefault('pdf_index', pdf_index)
                 meta.setdefault('page_number', i + 1) # Assuming cache stores pages in order
                 meta.setdefault('original_path', pdf_path)
                 meta.setdefault('cluster_id', None) # Reset cluster ID on load from cache
            # print(f"Cache hit for: {pdf_filename}") # Reduce verbosity
            return metadata_list, texts

        start_time = time.time()
        result_metadata = []
        result_texts = []
        pages = None # Initialize

        try:
            # Convert PDF to images (PIL Image objects)
            # Consider adding a timeout? pdf2image can hang on corrupted PDFs.
            # Use poppler_path argument if poppler is not in PATH
            pages = convert_from_path(pdf_path, dpi=300) # dpi=300 is good for OCR

            if not pages:
                 print(f"Warning: No pages found or converted for PDF {pdf_path}")
                 return None, None

            for i, page_image in enumerate(pages):
                # Initial metadata for this page
                metadata = {
                    'pdf_file': pdf_filename,
                    'pdf_index': pdf_index,
                    'page_number': i + 1,
                    'original_path': pdf_path,
                    'cluster_id': None # Initialize cluster ID
                }

                # Extract text and get updated metadata
                text, updated_metadata = self.extract_text_from_image(page_image, metadata)

                # Append results for this page
                result_metadata.append(updated_metadata)
                result_texts.append(text)

                # Optional: Close the PIL image object if not needed anymore to free memory
                page_image.close()

        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return None or partial results depending on desired robustness
            return None, None # Indicate failure for this PDF
        finally:
             # Explicitly clear the pages list to potentially free memory sooner
             if pages:
                  del pages

        # Calculate processing time for this PDF
        processing_time = time.time() - start_time
        self.file_times[pdf_filename] = processing_time
        # print(f"Processed {pdf_filename} in {processing_time:.2f} seconds")

        # Cache the results (metadata and text only)
        self.save_to_cache(pdf_filename, result_metadata, result_texts)

        return result_metadata, result_texts


    def process_all_documents(self):
        """Process all PDF files in parallel and aggregate results."""
        self.load_pdf_files()
        if not self.pdf_files:
             print("No PDFs to process. Exiting.")
             return # Exit if no files loaded

        # Reset global lists
        self.page_metadata = []
        self.page_texts = []
        self.file_times = {}

        # Start overall timer
        self.start_time = time.time()

        successful_pdfs = 0
        failed_pdfs = 0

        # Process PDFs in parallel
        # Using ProcessPoolExecutor for CPU-bound tasks (OCR, image processing)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all PDF processing tasks
            futures = {executor.submit(self.process_single_pdf, os.path.join(self.input_dir, pdf_file), i, pdf_file): pdf_file
                       for i, pdf_file in enumerate(self.pdf_files)}

            # Process results as they complete with a progress bar (tqdm)
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                pdf_filename = futures[future] # Get filename associated with this future
                try:
                    # Get results from the completed future
                    metadata_list, texts = future.result()

                    if metadata_list is not None and texts is not None:
                        # Successfully processed this PDF, append results
                        self.page_metadata.extend(metadata_list)
                        self.page_texts.extend(texts)
                        successful_pdfs += 1
                    else:
                        # Processing function indicated failure for this PDF
                        # Error message already printed in process_single_pdf
                        # print(f"Failed to process: {pdf_filename}")
                        failed_pdfs += 1

                except Exception as e:
                    # Catch any unexpected errors during future.result() or processing
                    print(f"Error getting result for {pdf_filename}: {e}")
                    failed_pdfs += 1


        # Calculate total processing time
        self.total_time = time.time() - self.start_time
        total_time_formatted = str(timedelta(seconds=int(self.total_time)))

        print("-" * 30)
        print(f"PDF Processing Summary:")
        print(f"  Successfully processed: {successful_pdfs} PDF(s)")
        print(f"  Failed to process:    {failed_pdfs} PDF(s)")
        print(f"  Total pages extracted: {len(self.page_metadata)}")
        print(f"  Total processing time: {total_time_formatted} (HH:MM:SS)")
        print("-" * 30)

        # Clean up page_images list as images are not stored globally anymore
        self.page_images = []


    def calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings using SequenceMatcher.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0 # Similarity is 0 if either string is empty

        # Use difflib's SequenceMatcher for ratio-based similarity
        # ratio() is relatively fast for moderately sized strings.
        return difflib.SequenceMatcher(None, text1, text2).ratio()


    def cluster_pages_by_content(self):
        """
        Cluster pages based on text content using TF-IDF and DBSCAN,
        applied separately for each detected language group.
        Handles pages without text or those classified as noise by DBSCAN.
        """
        if not self.page_metadata:
             print("No page data available to perform clustering.")
             return

        print("Clustering pages by content...")

        # Filter pages that have extracted text
        # Store tuple: (original_index, text, language)
        text_pages_data = [
            (i, self.page_texts[i], self.page_metadata[i].get('language', 'eng'))
            for i, meta in enumerate(self.page_metadata) if meta.get('has_text', False)
        ]

        if not text_pages_data:
            print("No text content found in any pages. Skipping text-based clustering.")
            # Assign all pages to a single 'no_text' cluster
            current_cluster_id = 0
            for i in range(len(self.page_metadata)):
                 self.page_metadata[i]['cluster_id'] = current_cluster_id
            print(f"Assigned all {len(self.page_metadata)} pages to cluster {current_cluster_id} (no text).")
            # Proceed to organize/export this single cluster
            self.organize_clusters()
            return


        # --- Language Grouping ---
        language_groups = defaultdict(list)
        # Map language -> list of (index_in_text_pages_data, original_page_index, text)
        for idx_in_tp, (original_idx, text, lang) in enumerate(text_pages_data):
            language_groups[lang].append((idx_in_tp, original_idx, text))

        # --- Clustering within each language group ---
        # Initialize cluster assignments for all pages that *have text*
        # Size is len(text_pages_data). Default to -1 (noise/unassigned).
        text_page_cluster_assignments = np.ones(len(text_pages_data), dtype=int) * -1
        current_global_cluster_id = 0

        for lang, lang_group_pages in language_groups.items():
            num_lang_pages = len(lang_group_pages)
            print(f"Processing {num_lang_pages} pages in language group: '{lang}'")

            # Need at least 2 pages to cluster
            if num_lang_pages < 2:
                print(f"  Skipping clustering for '{lang}': Only {num_lang_pages} page(s).")
                # Assign these pages to the noise cluster (-1) for now
                # The final assignment logic will handle them.
                continue

            # Extract data specific to this language group
            indices_in_text_pages = [data[0] for data in lang_group_pages]
            lang_texts = [data[2] for data in lang_group_pages]

            # Select appropriate stop words for TF-IDF
            stop_words = 'english' # Default
            # Use ISO 639-1 codes if available and supported by sklearn
            lang_code_short = lang[:2] # e.g., 'en', 'es', 'fr'
            supported_stopwords = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'} # Add more
            if lang_code_short in supported_stopwords:
                 stop_words = supported_stopwords[lang_code_short]
                 # print(f"  Using '{stop_words}' stop words for language '{lang}'.")


            try:
                 # --- TF-IDF Vectorization ---
                 # Adjust max_features based on number of pages?
                 max_features = min(2000, max(100, num_lang_pages * 5))
                 # Consider min_df, max_df parameters
                 vectorizer = TfidfVectorizer(max_features=max_features,
                                             stop_words=stop_words,
                                             min_df=2 if num_lang_pages > 10 else 1, # Ignore terms in only 1 doc if many pages
                                             max_df=0.95) # Ignore terms in >95% of docs
                 tfidf_matrix = vectorizer.fit_transform(lang_texts)

                 # --- DBSCAN Clustering ---
                 # Parameters are crucial and data-dependent. Need tuning.
                 # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                 # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
                 min_samples = max(2, min(5, num_lang_pages // 10)) # Heuristic: 2-5 samples, scales slightly with size
                 eps = 0.6 # Cosine distance threshold (0=identical, 1=orthogonal). Lower means more clusters. Try 0.4-0.7.

                 # print(f"  Running DBSCAN with eps={eps}, min_samples={min_samples}") # Reduce verbosity
                 dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1) # n_jobs=-1 uses all cores
                 lang_specific_clusters = dbscan.fit_predict(tfidf_matrix)

                 # --- Map local cluster IDs to global ones ---
                 num_clusters_found = len(set(lang_specific_clusters)) - (1 if -1 in lang_specific_clusters else 0)
                 print(f"  Found {num_clusters_found} cluster(s) and {np.sum(lang_specific_clusters == -1)} noise point(s) in '{lang}'.")

                 # Create a mapping for non-noise clusters
                 local_to_global_id_map = {}
                 for local_id in np.unique(lang_specific_clusters):
                     if local_id != -1: # Not noise
                         local_to_global_id_map[local_id] = current_global_cluster_id
                         current_global_cluster_id += 1

                 # Apply the mapping to the results for this language group
                 for i, local_cluster_id in enumerate(lang_specific_clusters):
                     idx_in_text_pages = indices_in_text_pages[i] # Get the index in the main text_pages list
                     if local_cluster_id != -1:
                         # Assign the unique global ID
                         text_page_cluster_assignments[idx_in_text_pages] = local_to_global_id_map[local_cluster_id]
                     # else: it remains -1 (noise)

            except ValueError as ve:
                 print(f"  Skipping clustering for language '{lang}' due to error: {ve}")
                 # Assign these pages to noise (-1), will be handled later
            except Exception as e:
                 print(f"  Error during clustering for language '{lang}': {e}")
                 # Assign these pages to noise (-1)

        # --- Final Cluster Assignment ---
        # Now, map the cluster assignments back to the original self.page_metadata list
        # And handle pages that had no text or were marked as noise by DBSCAN.

        # Assign a dedicated cluster ID for noise/no_text pages
        noise_cluster_id = current_global_cluster_id
        assigned_noise_cluster = False

        for i, meta in enumerate(self.page_metadata):
             if meta.get('has_text', False):
                  # This page had text, find its assignment from text_page_cluster_assignments
                  try:
                       # Find the index of this original page in the text_pages_data list
                       text_page_index = -1
                       for idx_tp, (orig_idx, _, _) in enumerate(text_pages_data):
                            if orig_idx == i:
                                 text_page_index = idx_tp
                                 break

                       if text_page_index != -1:
                            cluster_assignment = text_page_cluster_assignments[text_page_index]
                            if cluster_assignment == -1: # Marked as noise by DBSCAN
                                 meta['cluster_id'] = noise_cluster_id
                                 assigned_noise_cluster = True
                            else: # Assigned to a valid cluster
                                 meta['cluster_id'] = int(cluster_assignment) # Ensure it's int
                       else:
                            # Should not happen if has_text is true, but safety check
                            print(f"Warning: Could not find text page index for original index {i}. Assigning to noise.")
                            meta['cluster_id'] = noise_cluster_id
                            assigned_noise_cluster = True

                  except Exception as find_e:
                       print(f"Error finding text page index for page {i}: {find_e}. Assigning to noise.")
                       meta['cluster_id'] = noise_cluster_id
                       assigned_noise_cluster = True
             else:
                  # This page had no text originally
                  meta['cluster_id'] = noise_cluster_id
                  assigned_noise_cluster = True

        if assigned_noise_cluster:
             print(f"Assigned noise / no-text pages to cluster ID: {noise_cluster_id}")
             # Optional: Increment the global ID counter if this noise cluster ID is used
             current_global_cluster_id += 1 # Make sure noise ID doesn't clash

        # --- Post-Clustering Steps ---
        # Check for document continuity (optional but recommended)
        self.check_document_continuity()

        # Organize pages into final cluster structures
        self.organize_clusters()


    def check_document_continuity(self):
        """
        Refine clusters by merging adjacent pages within the same PDF if they
        were assigned to different clusters but seem logically connected.
        Considers language and text similarity.
        """
        if not self.page_metadata: return
        print("Checking document continuity...")

        # Sort page indices based on PDF index and page number
        sorted_page_indices = sorted(range(len(self.page_metadata)),
                                     key=lambda k: (self.page_metadata[k]['pdf_index'], self.page_metadata[k]['page_number']))

        # --- Build Merge Map ---
        # This map stores potential merges: source_cluster_id -> target_cluster_id
        # We aim to map higher cluster IDs to lower ones for consistency.
        merge_map = {}

        for i in range(len(sorted_page_indices) - 1):
            idx1 = sorted_page_indices[i]
            idx2 = sorted_page_indices[i+1]

            meta1 = self.page_metadata[idx1]
            meta2 = self.page_metadata[idx2]

            # 1. Check if pages are from the same PDF and consecutive
            if meta1['pdf_index'] != meta2['pdf_index'] or \
               meta2['page_number'] != meta1['page_number'] + 1:
                continue

            # 2. Get cluster IDs, skip if either is None or they are already the same
            c1 = meta1.get('cluster_id')
            c2 = meta2.get('cluster_id')
            if c1 is None or c2 is None or c1 == c2:
                continue

            # --- Merge Decision Logic ---
            should_merge = False

            # 3. Language Check: Avoid merging known different languages
            lang1 = meta1.get('language', 'unknown')
            lang2 = meta2.get('language', 'unknown')
            if lang1 != 'unknown' and lang2 != 'unknown' and lang1 != lang2:
                continue # Don't merge known different languages

            # 4. Text Similarity Check (if both pages have text)
            text1 = self.page_texts[idx1] if meta1.get('has_text') else ""
            text2 = self.page_texts[idx2] if meta2.get('has_text') else ""

            if text1 and text2:
                # Compare end of text1 with start of text2
                similarity_threshold = 0.20 # Tune this threshold (0.1-0.3 maybe)
                text_end = text1[-150:] # Look at last 150 chars
                text_start = text2[:150] # Look at first 150 chars
                text_similarity = self.calculate_text_similarity(text_end, text_start)

                if text_similarity >= similarity_threshold:
                    should_merge = True
                    # print(f"Merging C{c1} & C{c2} (pg {meta1['page_number']},{meta2['page_number']} of {meta1['pdf_file']}) based on text similarity {text_similarity:.2f}")

            # 5. Heuristic: Merge if adjacent and languages match (or are unknown),
            #    even without text similarity? Can be risky. Enable cautiously.
            # Example: If languages are compatible AND maybe doc_type is similar?
            # if not should_merge and (lang1 == lang2 or lang1 == 'unknown' or lang2 == 'unknown'):
            #     # Add more conditions here if desired, e.g., check document type consistency
            #     # doc_type1 = meta1.get('doc_type', 'unknown')
            #     # doc_type2 = meta2.get('doc_type', 'unknown')
            #     # if doc_type1 == doc_type2 and doc_type1 != 'unknown':
            #     #    should_merge = True
            #     #    print(f"Merging C{c1} & C{c2} based on sequence and compatible language/type.")
            #     pass


            # --- Register the Merge ---
            if should_merge:
                # Always merge the higher cluster ID into the lower one
                source_cluster = max(c1, c2)
                target_cluster = min(c1, c2)

                # Add to map, handling potential chains (if source is already merging)
                # Find the ultimate target for the source cluster
                while source_cluster in merge_map:
                    source_cluster = merge_map[source_cluster]

                # Find the ultimate target for the current target cluster
                while target_cluster in merge_map:
                     target_cluster = merge_map[target_cluster]

                # If the ultimate targets are different, create/update the merge link
                if source_cluster != target_cluster:
                     final_source = max(source_cluster, target_cluster)
                     final_target = min(source_cluster, target_cluster)
                     # Ensure we don't create a loop by mapping back to the original source
                     if final_source != final_target:
                          merge_map[final_source] = final_target


        # --- Apply Merges ---
        if merge_map:
            print(f"Applying {len(merge_map)} document continuity merges...")
            # Build final mapping (resolve multi-step chains)
            final_mapping = {}
            # Use list to make a copy for safe iteration if merge_map is modified
            source_clusters_to_process = list(merge_map.keys())

            for source_cluster in source_clusters_to_process:
                # Skip if already processed as part of another chain
                if source_cluster in final_mapping:
                     continue

                target = merge_map[source_cluster]
                path = [source_cluster, target] # Track path to detect loops
                # Follow the chain until the ultimate target is found
                while target in merge_map:
                    target = merge_map[target]
                    if target in path:
                         print(f"Warning: Detected merge loop involving cluster {target}. Breaking loop.")
                         # Break the loop: don't map the original source in this case.
                         # Or choose a strategy like keeping the lowest ID in the loop.
                         target = min(path) # Example: map all in loop to the minimum ID
                         for node in path:
                              if node != target:
                                   final_mapping[node] = target
                         break # Exit inner while loop for this chain
                    path.append(target)
                else: # Normal termination (no loop detected for this chain)
                     # Map all nodes in the path (except the final target) to the final target
                     final_target = path[-1]
                     for node in path[:-1]:
                          final_mapping[node] = final_target


            # Apply the final mapping to all pages
            applied_count = 0
            if final_mapping: # Check if any mappings were actually created
                 for i, meta in enumerate(self.page_metadata):
                     original_cluster_id = meta.get('cluster_id')
                     if original_cluster_id is not None and original_cluster_id in final_mapping:
                          final_target_id = final_mapping[original_cluster_id]
                          # Assign the final target ID if it's different
                          if meta['cluster_id'] != final_target_id:
                               self.page_metadata[i]['cluster_id'] = final_target_id
                               applied_count += 1
                 print(f"Remapped cluster IDs for {applied_count} pages.")
            else:
                 print("No effective merges after resolving chains.")
        else:
             print("No document continuity merges needed.")


    def organize_clusters(self):
        """
        Group pages into final document cluster structures based on their
        assigned cluster IDs after continuity checks.
        """
        if not self.page_metadata: return
        print("Organizing document clusters...")

        # Group page indices by their final cluster ID
        clusters_by_id = defaultdict(list)
        for i, meta in enumerate(self.page_metadata):
            cluster_id = meta.get('cluster_id')
            if cluster_id is not None: # Include noise cluster ID if present
                clusters_by_id[cluster_id].append(i)
            else:
                # Handle pages somehow missed (shouldn't happen ideally)
                 clusters_by_id['unassigned'].append(i)


        # Create final document cluster objects
        self.document_clusters = []
        processed_ids = set()

        # Process clusters by ID (sort for consistent output)
        # Treat non-integer IDs (like 'noise', 'unassigned') separately
        numeric_ids = sorted([cid for cid in clusters_by_id if isinstance(cid, int)])
        other_ids = sorted([cid for cid in clusters_by_id if not isinstance(cid, int)])


        for cluster_id in numeric_ids + other_ids:
            if cluster_id in processed_ids: continue

            page_indices = clusters_by_id[cluster_id]
            processed_ids.add(cluster_id)

            if not page_indices: continue

            # Get metadata for pages in this cluster
            cluster_pages_meta = [self.page_metadata[i] for i in page_indices]

            # Sort pages within the cluster by original PDF and page number
            cluster_pages_meta.sort(key=lambda x: (x['pdf_index'], x['page_number']))

            # --- Determine consolidated properties for the cluster ---
            # Document Type: Majority vote among known types
            doc_types = [p.get('doc_type', 'unknown') for p in cluster_pages_meta]
            known_doc_types = [dt for dt in doc_types if dt != 'unknown']
            if known_doc_types:
                cluster_doc_type = max(set(known_doc_types), key=known_doc_types.count)
            else:
                # Use cluster ID if it's descriptive (like 'noise', 'unassigned')
                # If the cluster ID is numeric but all pages were unknown, still 'unknown'
                cluster_doc_type = str(cluster_id) if not isinstance(cluster_id, int) else 'unknown'

            # Language: Majority vote among known languages
            languages = [p.get('language', 'unknown') for p in cluster_pages_meta]
            known_languages = [lang for lang in languages if lang != 'unknown']
            if known_languages:
                cluster_language = max(set(known_languages), key=known_languages.count)
            else:
                cluster_language = 'unknown'

            # List of unique PDF files involved
            cluster_pdf_files = sorted(list(set(p['pdf_file'] for p in cluster_pages_meta)))

            # --- Create cluster object ---
            cluster = {
                'cluster_id': cluster_id,
                'page_count': len(page_indices),
                'document_type': cluster_doc_type, # Consolidated type
                'language': cluster_language,   # Consolidated language
                'pages': cluster_pages_meta,   # List of individual page metadata dicts
                'pdf_files': cluster_pdf_files # List of unique source PDF filenames
            }

            self.document_clusters.append(cluster)


        # Sort final clusters (e.g., by cluster ID, then page count)
        # Make sure numeric IDs come before string IDs like 'noise'
        self.document_clusters.sort(key=lambda x: (isinstance(x['cluster_id'], str), x['cluster_id']))


        print(f"Organized into {len(self.document_clusters)} final document clusters.")


    def export_results(self):
        """
        Export clustering results: summary CSV, detailed page CSV,
        and organize text files into cluster-specific directories using symlinks.
        """
        if not self.document_clusters:
             print("No clusters to export.")
             return None # Indicate nothing was exported

        print("Exporting results...")
        export_base_dir = self.output_dir
        cluster_details_dir = os.path.join(export_base_dir, "document_clusters") # Dir for symlinks

        # --- 1. Cluster Summary CSV ---
        cluster_summary_data = []
        for cluster in self.document_clusters:
            first_page_meta = cluster['pages'][0] if cluster['pages'] else {}
            cluster_summary_data.append({
                'cluster_id': cluster['cluster_id'],
                'document_type': cluster['document_type'],
                'language': cluster.get('language', 'unknown'),
                'page_count': cluster['page_count'],
                'pdf_files_involved': ', '.join(cluster['pdf_files']),
                'first_page_example': f"{first_page_meta.get('pdf_file','NA')}_p{first_page_meta.get('page_number','NA')}"
            })

        summary_df = pd.DataFrame(cluster_summary_data)
        summary_path = os.path.join(export_base_dir, "cluster_summary.csv")
        try:
            summary_df.to_csv(summary_path, index=False)
            print(f"  Cluster summary saved to: {summary_path}")
        except Exception as e:
            print(f"  Error saving cluster summary CSV: {e}")
            summary_path = None # Mark as failed


        # --- 2. Language Statistics CSV (from summary) ---
        if not summary_df.empty and 'language' in summary_df.columns:
             try:
                  language_stats = summary_df['language'].value_counts().reset_index()
                  language_stats.columns = ['language', 'cluster_count']
                  language_stats_path = os.path.join(export_base_dir, "language_statistics.csv")
                  language_stats.to_csv(language_stats_path, index=False)
                  print(f"  Language statistics saved to: {language_stats_path}")
             except Exception as e:
                  print(f"  Error saving language statistics: {e}")
        else:
             print("  Skipping language statistics (no language data or summary failed).")


        # --- 3. Detailed Page Information CSV ---
        page_details_data = []
        for cluster in self.document_clusters:
            cluster_id = cluster['cluster_id']
            for page_meta in cluster['pages']:
                 # Ensure paths exist before trying relpath
                 enhanced_img_relpath = ''
                 if page_meta.get('enhanced_image_path') and os.path.exists(page_meta['enhanced_image_path']):
                     try:
                          enhanced_img_relpath = os.path.relpath(page_meta['enhanced_image_path'], export_base_dir)
                     except ValueError: # Handles case where paths are on different drives (Windows)
                          enhanced_img_relpath = page_meta['enhanced_image_path'] # Use absolute path as fallback

                 text_relpath = ''
                 if page_meta.get('text_path') and os.path.exists(page_meta['text_path']):
                     try:
                          text_relpath = os.path.relpath(page_meta['text_path'], export_base_dir)
                     except ValueError:
                          text_relpath = page_meta['text_path']


                 page_details_data.append({
                    'cluster_id': cluster_id,
                    'pdf_file': page_meta['pdf_file'],
                    'page_number': page_meta['page_number'],
                    'doc_type': page_meta.get('doc_type', 'unknown'),
                    'language': page_meta.get('language', 'unknown'),
                    'text_length': page_meta.get('text_length', 0),
                    'has_text': page_meta.get('has_text', False),
                    'enhanced_image_path': enhanced_img_relpath,
                    'text_path': text_relpath
                 })

        if page_details_data:
             detail_df = pd.DataFrame(page_details_data)
             # Save directly in the main output directory
             detail_path = os.path.join(export_base_dir, "cluster_page_details.csv")
             try:
                  detail_df.to_csv(detail_path, index=False)
                  print(f"  Detailed page information saved to: {detail_path}")
             except Exception as e:
                  print(f"  Error saving detailed page CSV: {e}")
        else:
             print("  Skipping detailed page information (no page data).")


        # --- 4. Organize Text Files/Images by Cluster (using Symlinks) ---
        print("  Organizing output files by cluster (using symlinks)...")
        # Base directory for clustered links was created earlier
        # os.makedirs(cluster_details_dir, exist_ok=True) # Ensure it exists

        link_count = 0
        link_errors = 0
        for cluster in self.document_clusters:
             cluster_id_str = str(cluster['cluster_id']).replace(' ', '_') # Sanitize ID for dir name
             target_cluster_dir = os.path.join(cluster_details_dir, f"cluster_{cluster_id_str}")
             os.makedirs(target_cluster_dir, exist_ok=True)

             for page_meta in cluster['pages']:
                  # Link Text File
                  text_path_abs = page_meta.get('text_path')
                  if text_path_abs and os.path.exists(text_path_abs):
                       link_name_txt = f"{page_meta['pdf_file']}_p{page_meta['page_number']}.txt"
                       link_path_txt = os.path.join(target_cluster_dir, link_name_txt)
                       try:
                            # Create relative symlink from link location to target
                            link_dir = os.path.dirname(link_path_txt)
                            target_relpath = os.path.relpath(text_path_abs, start=link_dir)

                            if os.path.lexists(link_path_txt): # Use lexists for links
                                 os.remove(link_path_txt)
                            os.symlink(target_relpath, link_path_txt)
                            link_count += 1
                       except OSError as oe:
                            # On Windows, symlinks might require admin privileges or dev mode
                            # Fallback to copying the file? Or just warn.
                            # print(f"Warning: Symlink creation failed for text {link_name_txt} (may require admin rights on Windows): {oe}")
                            link_errors += 1
                       except Exception as e:
                            print(f"Warning: Could not create symlink for text {link_name_txt} in cluster {cluster_id_str}: {e}")
                            link_errors += 1

                  # Link Enhanced Image File (Optional)
                  image_path_abs = page_meta.get('enhanced_image_path')
                  if image_path_abs and os.path.exists(image_path_abs):
                       link_name_img = f"{page_meta['pdf_file']}_p{page_meta['page_number']}.png"
                       link_path_img = os.path.join(target_cluster_dir, link_name_img)
                       try:
                            link_dir = os.path.dirname(link_path_img)
                            target_relpath = os.path.relpath(image_path_abs, start=link_dir)

                            if os.path.lexists(link_path_img):
                                 os.remove(link_path_img)
                            os.symlink(target_relpath, link_path_img)
                            # link_count += 1 # Count image links separately if needed
                       except OSError as oe:
                            # print(f"Warning: Symlink creation failed for image {link_name_img} (may require admin rights on Windows): {oe}")
                            link_errors += 1
                       except Exception as e:
                            print(f"Warning: Could not create symlink for image {link_name_img} in cluster {cluster_id_str}: {e}")
                            link_errors += 1

        print(f"  Created/updated {link_count} text file symlinks with {link_errors} errors.")
        print("File organization complete.")

        return summary_path # Return path to summary file, or None if failed


# --- Main execution block ---
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Optimized PDF Document Clustering")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing PDF files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for results")
    parser.add_argument("-s", "--sample", type=int, default=None, help="Number of PDFs to process (default: all)")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Max worker processes (default: CPU count - 1)")
    parser.add_argument("-n", "--no-cache", action="store_true", help="Disable caching of intermediate results")
    parser.add_argument("-c", "--clear-cache", action="store_true", help="Clear existing cache before processing")

    args = parser.parse_args()

    # --- Cache Handling ---
    if args.clear_cache:
        cache_dir = os.path.join(args.output, "cache")
        if os.path.exists(cache_dir):
            try:
                print(f"Clearing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
                # No need to recreate cache dir immediately, _create_directories will do it
            except Exception as e:
                 print(f"Warning: Could not clear cache directory {cache_dir}: {e}")


    # --- Initialize and Run Clustering ---
    # Record start time
    overall_start_time = time.time()

    # Instantiate the main class
    clustering = OptimizedDocumentClustering(
        input_dir=args.input,
        output_dir=args.output,
        sample_size=args.sample,
        max_workers=args.workers,
        use_cache=not args.no_cache # use_cache is True if --no-cache is *not* present
    )

    # --- Execute Workflow ---
    # 1. Process PDFs (Convert to text/metadata)
    clustering.process_all_documents()

    # 2. Cluster pages (only if pages were successfully processed)
    if clustering.page_metadata:
         # 2a. Cluster based on content (handles continuity and noise)
         clustering.cluster_pages_by_content()
         # 2b. Organize results into final cluster structures (already called within cluster_pages_by_content)
         # clustering.organize_clusters() # No longer needed here

         # 3. Export results (summary, details, links)
         clustering.export_results()
    else:
         print("No pages were processed successfully. Skipping clustering and export.")


    # --- Finish ---
    overall_end_time = time.time()
    print("-" * 30)
    print(f"Document clustering process finished.")
    print(f"Total execution time: {str(timedelta(seconds=int(overall_end_time - overall_start_time)))} (HH:MM:SS)")
    print(f"Results exported to: {args.output}")
    print("-" * 30)