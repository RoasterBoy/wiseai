import os
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import difflib
import re
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import argparse
import time
from datetime import timedelta
import multiprocessing
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentClusteringOptimized:
    def __init__(self, input_dir, output_dir, sample_size=None, 
                 save_enhanced_images=False, save_text_output=True, 
                 max_workers=None):
        """
        Initialize the document clustering prototype with optimized parameters.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save results
            sample_size: Number of PDFs to process (None for all)
            save_enhanced_images: Whether to save preprocessed images (defaults to False)
            save_text_output: Whether to save extracted text files (defaults to True)
            max_workers: Maximum number of worker processes (defaults to CPU count)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.pdf_files = []
        self.page_metadata = []
        self.page_texts = []
        self.document_clusters = []
        
        # Configuration flags
        self.save_enhanced_images = save_enhanced_images
        self.save_text_output = save_text_output
        self.max_workers = max_workers if max_workers else min(os.cpu_count(), 4)
        
        # Timing information
        self.start_time = None
        self.file_times = {}
        self.total_time = 0
        
        # Create only necessary output directories
        os.makedirs(output_dir, exist_ok=True)
        
        if self.save_enhanced_images:
            os.makedirs(os.path.join(output_dir, "enhanced_images"), exist_ok=True)
        
        if self.save_text_output:
            os.makedirs(os.path.join(output_dir, "text_output"), exist_ok=True)
            
        os.makedirs(os.path.join(output_dir, "document_clusters"), exist_ok=True)
        
        logger.info(f"Initialized DocumentClusteringOptimized with {self.max_workers} workers")
        
    def load_pdf_files(self):
        """Load PDF files from the input directory."""
        self.pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        
        if self.sample_size:
            self.pdf_files = self.pdf_files[:min(self.sample_size, len(self.pdf_files))]
            
        logger.info(f"Found {len(self.pdf_files)} PDF files to process")
        
    def convert_pdfs_to_images(self, pdf_path, pdf_index):
        """
        Convert a PDF file to images.
        
        Args:
            pdf_path: Path to the PDF file
            pdf_index: Index of the PDF in the list of files
            
        Returns:
            List of (image, metadata) tuples
        """
        result = []
        try:
            # Only log when not in a worker process to avoid cluttered output
            if multiprocessing.current_process().name == 'MainProcess':
                logger.info(f"Converting PDF {pdf_index+1}/{len(self.pdf_files)}: {pdf_path}")
            
            # Convert PDF to images with optimized parameters
            pages = convert_from_path(
                pdf_path, 
                dpi=300,
                thread_count=2,  # Use 2 threads per PDF for conversion
                use_pdftocairo=True,  # pdftocairo is generally faster than pdftoppm
                grayscale=True  # Convert directly to grayscale to save memory
            )
            
            for i, page in enumerate(pages):
                metadata = {
                    'pdf_file': os.path.basename(pdf_path),
                    'pdf_index': pdf_index,
                    'page_number': i + 1,
                    'original_path': pdf_path,
                    'cluster_id': None
                }
                result.append((page, metadata))
                
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
            
        return result
    
    def preprocess_image(self, image):
        """
        Optimized image preprocessing for OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image object
        """
        # Convert to NumPy array for OpenCV processing
        opencv_image = np.array(image)
        
        # If image is RGB, convert to grayscale
        if len(opencv_image.shape) == 3:
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = opencv_image
        
        # Apply bilateral filter for noise reduction while preserving edges
        # This is more effective than simple blurring for OCR
        denoised = cv2.bilateralFilter(gray, 5, 75, 75)
        
        # Use adaptive thresholding instead of global thresholding
        # This handles different lighting conditions within the image
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Optional: Apply slight morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL for Tesseract compatibility
        return Image.fromarray(cleaned)
    
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
        
        # Count occurrences
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
            image_filename = f"{metadata['pdf_index']}_{metadata['page_number']}.png"
            enhanced_image_path = os.path.join(self.output_dir, "enhanced_images", image_filename)
            enhanced_image.save(enhanced_image_path)
            metadata['enhanced_image_path'] = enhanced_image_path
        else:
            metadata['enhanced_image_path'] = None
        
        # Extract text using Tesseract OCR with optimized config
        try:
            # Use multi-language mode directly - faster than running OCR 3 times
            custom_config = r'--oem 1 --psm 3 -l eng+spa+fra'
            text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Detect language from the extracted text
            detected_lang = self.detect_language(text)
            
            # Save extracted text if enabled
            if self.save_text_output:
                text_filename = f"{metadata['pdf_index']}_{metadata['page_number']}.txt"
                text_path = os.path.join(self.output_dir, "text_output", text_filename)
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
                metadata['text_path'] = text_path
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
            metadata['text_path'] = None
            metadata['text_length'] = 0
            metadata['has_text'] = False
            metadata['language'] = 'unknown'
            metadata['doc_type'] = 'unknown'
            return "", metadata
            
    def detect_document_type(self, text, language='eng'):
        """
        Detect document type based on text patterns, with multilingual support.
        
        Args:
            text: Extracted text
            language: Detected language code ('eng', 'spa', 'fra')
            
        Returns:
            Document type string
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
    
    def process_single_pdf(self, pdf_index, pdf_file):
        """
        Process a single PDF file and return results.
        
        Args:
            pdf_index: Index of the PDF in the file list
            pdf_file: Name of the PDF file
            
        Returns:
            Tuple of (page_metadata, page_texts, duration)
        """
        start_time = time.time()
        
        pdf_path = os.path.join(self.input_dir, pdf_file)
        
        if multiprocessing.current_process().name == 'MainProcess':
            logger.info(f"Processing {pdf_index+1}/{len(self.pdf_files)}: {pdf_file}")
        
        page_metadata = []
        page_texts = []
        
        try:
            # Convert PDF to images
            page_results = self.convert_pdfs_to_images(pdf_path, pdf_index)
            
            # Process each page
            for image, metadata in page_results:
                text, updated_metadata = self.extract_text_from_image(image, metadata)
                
                page_metadata.append(updated_metadata)
                page_texts.append(text)
                
        except Exception as e:
            logger.error(f"Error in processing PDF {pdf_file}: {e}")
            
        duration = time.time() - start_time
        return page_metadata, page_texts, duration
    
    def process_all_documents(self):
        """Process all PDF files and extract text in parallel."""
        self.load_pdf_files()
        
        # Start overall timer
        self.start_time = time.time()
        
        all_page_metadata = []
        all_page_texts = []
        
        # Process PDFs in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create processing tasks
            future_to_pdf = {
                executor.submit(self.process_single_pdf, i, pdf_file): (i, pdf_file) 
                for i, pdf_file in enumerate(self.pdf_files)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_pdf):
                i, pdf_file = future_to_pdf[future]
                try:
                    pdf_metadata, pdf_texts, duration = future.result()
                    all_page_metadata.extend(pdf_metadata)
                    all_page_texts.extend(pdf_texts)
                    
                    # Store and print the duration
                    self.file_times[pdf_file] = duration
                    duration_formatted = str(timedelta(seconds=int(duration)))
                    logger.info(f"Completed {pdf_file} ({i+1}/{len(self.pdf_files)}) in {duration_formatted}")
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
        
        self.page_metadata = all_page_metadata
        self.page_texts = all_page_texts
        
        # Calculate total processing time
        self.total_time = time.time() - self.start_time
        total_time_formatted = str(timedelta(seconds=int(self.total_time)))
        logger.info(f"Total processing time: {total_time_formatted}")
        logger.info(f"Processed {len(self.page_metadata)} pages from {len(self.pdf_files)} PDF files")
    
    def calculate_text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
            
        # For short texts, use SequenceMatcher
        if len(text1) < 500 and len(text2) < 500:
            return difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # For longer texts, use a faster approach with word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def cluster_pages_by_content(self):
        """
        Cluster pages based on text content with optimized processing.
        """
        logger.info("Clustering pages by content...")
        
        # Filter pages with text
        text_pages = [(i, text, self.page_metadata[i].get('language', 'eng')) 
                      for i, text in enumerate(self.page_texts) if text and text.strip()]
        
        if not text_pages:
            logger.warning("No text content found in any pages")
            return
            
        # Group pages by language for more accurate clustering
        language_groups = {}
        for i, (page_idx, text, lang) in enumerate(text_pages):
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append((i, page_idx, text))
        
        # Initialize all pages as noise
        all_clusters = [-1] * len(self.page_metadata)
        next_cluster_id = 0
        
        # Process each language group separately
        for lang, lang_pages in language_groups.items():
            if len(lang_pages) < 2:  # Skip languages with only one page
                continue
                
            logger.info(f"Clustering {len(lang_pages)} pages in language: {lang}")
            
            lang_indices = [i for i, _, _ in lang_pages]
            lang_page_indices = [page_idx for _, page_idx, _ in lang_pages]
            lang_texts = [text for _, _, text in lang_pages]
            
            # Select appropriate stop words based on language
            stop_words = 'english'
            if lang == 'spa':
                stop_words = 'spanish'
            elif lang == 'fra':
                stop_words = 'french'
            
            try:
                # Create TF-IDF vectors with optimized parameters
                vectorizer = TfidfVectorizer(
                    max_features=500,
                    stop_words=stop_words,
                    min_df=2,         # Ignore terms that appear in fewer than 2 docs
                    max_df=0.9,       # Ignore terms that appear in more than 90% of docs
                    ngram_range=(1,2) # Include both unigrams and bigrams
                )
                
                # Process in batches if the dataset is large
                if len(lang_texts) > 1000:
                    # Implement batch processing for large datasets
                    batch_size = 500
                    batches = (len(lang_texts) + batch_size - 1) // batch_size
                    
                    # Process all batches
                    for b in range(batches):
                        start_idx = b * batch_size
                        end_idx = min((b + 1) * batch_size, len(lang_texts))
                        
                        batch_texts = lang_texts[start_idx:end_idx]
                        batch_indices = lang_page_indices[start_idx:end_idx]
                        
                        # Process this batch
                        tfidf_matrix = vectorizer.fit_transform(batch_texts)
                        
                        # Calculate distances
                        distances = cosine_distances(tfidf_matrix)
                        
                        # Cluster with DBSCAN
                        dbscan = DBSCAN(
                            eps=0.5,
                            min_samples=2,
                            metric='precomputed'
                        )
                        batch_clusters = dbscan.fit_predict(distances)
                        
                        # Map cluster IDs
                        for i, cluster_id in enumerate(batch_clusters):
                            if cluster_id >= 0:  # Not noise
                                page_idx = batch_indices[i]
                                all_clusters[page_idx] = cluster_id + next_cluster_id
                        
                        # Update next_cluster_id
                        if batch_clusters.max() >= 0:
                            next_cluster_id += batch_clusters.max() + 1
                else:
                    # Process all at once for smaller datasets
                    tfidf_matrix = vectorizer.fit_transform(lang_texts)
                    
                    # Calculate cosine distances
                    distances = cosine_distances(tfidf_matrix)
                    
                    # Cluster using DBSCAN with precomputed distances
                    dbscan = DBSCAN(
                        eps=0.5,
                        min_samples=2,
                        metric='precomputed'
                    )
                    lang_clusters = dbscan.fit_predict(distances)
                    
                    # Map cluster IDs to global cluster IDs
                    for i, cluster_id in enumerate(lang_clusters):
                        if cluster_id >= 0:  # Not noise
                            page_idx = lang_page_indices[i]
                            all_clusters[page_idx] = cluster_id + next_cluster_id
                    
                    # Update next_cluster_id
                    if lang_clusters.max() >= 0:
                        next_cluster_id += lang_clusters.max() + 1
                        
            except Exception as e:
                logger.error(f"Error clustering language {lang}: {e}")
        
        # Update metadata with cluster IDs
        for idx, cluster_id in enumerate(all_clusters):
            if idx < len(self.page_metadata):
                self.page_metadata[idx]['cluster_id'] = int(cluster_id) if cluster_id >= 0 else None
        
        # Handle pages with no text
        self.process_pages_without_text()
        
        # Check for document continuity
        self.check_document_continuity()
        
        # Organize results
        self.organize_clusters()
    
    def process_pages_without_text(self):
        """
        Process pages that have no text or were not clustered.
        """
        logger.info("Processing pages without text...")
        
        # Group unclustered pages
        unclustered_indices = [i for i, meta in enumerate(self.page_metadata) 
                              if meta['cluster_id'] is None]
        
        if not unclustered_indices:
            return
            
        logger.info(f"Found {len(unclustered_indices)} unclustered pages")
        
        # Get maximum cluster ID used so far
        next_cluster_id = max([meta['cluster_id'] for meta in self.page_metadata 
                              if meta['cluster_id'] is not None], default=-1) + 1
        
        # Group unclustered pages by PDF file
        pdf_groups = {}
        for idx in unclustered_indices:
            pdf_file = self.page_metadata[idx]['pdf_file']
            if pdf_file not in pdf_groups:
                pdf_groups[pdf_file] = []
            pdf_groups[pdf_file].append(idx)
        
        # Process each PDF file group
        for pdf_file, indices in pdf_groups.items():
            # Sort pages by page number
            sorted_indices = sorted(indices, key=lambda idx: self.page_metadata[idx]['page_number'])
            
            # Find consecutive pages and assign them to the same cluster
            current_cluster = None
            prev_page_num = None
            
            for idx in sorted_indices:
                page_num = self.page_metadata[idx]['page_number']
                
                # If this page is consecutive with the previous one, assign same cluster
                if prev_page_num is not None and page_num == prev_page_num + 1:
                    self.page_metadata[idx]['cluster_id'] = current_cluster
                else:
                    # Start a new cluster
                    current_cluster = next_cluster_id
                    next_cluster_id += 1
                    self.page_metadata[idx]['cluster_id'] = current_cluster
                
                prev_page_num = page_num
    
    def check_document_continuity(self):
        """
        Check for document continuity based on page numbers and text flow.
        Merge clusters that are likely part of the same document.
        """
        logger.info("Checking document continuity...")
        
        # Sort pages by PDF and page number
        sorted_pages = sorted(enumerate(self.page_metadata), 
                             key=lambda x: (x[1]['pdf_index'], x[1]['page_number']))
        
        # Check for sequential pages with different clusters
        merges = []
        
        for i in range(len(sorted_pages) - 1):
            idx1, meta1 = sorted_pages[i]
            idx2, meta2 = sorted_pages[i + 1]
            
            # Only check pages in the same PDF
            if meta1['pdf_index'] != meta2['pdf_index']:
                continue
                
            # Only check if pages are consecutive
            if meta2['page_number'] != meta1['page_number'] + 1:
                continue
                
            # Skip if either page doesn't have a cluster
            if meta1['cluster_id'] is None or meta2['cluster_id'] is None:
                continue
                
            # Skip if pages are already in the same cluster
            if meta1['cluster_id'] == meta2['cluster_id']:
                continue
                
            # Check if pages are in the same language
            lang1 = meta1.get('language', 'unknown')
            lang2 = meta2.get('language', 'unknown')
            
            # If languages are different and known, don't merge
            if lang1 != 'unknown' and lang2 != 'unknown' and lang1 != lang2:
                continue
                
            # Get text for both pages
            text1 = self.page_texts[idx1] if idx1 < len(self.page_texts) else ""
            text2 = self.page_texts[idx2] if idx2 < len(self.page_texts) else ""
            
            # Calculate text continuity for the last/first 100 chars
            text_similarity = 0
            if text1 and text2:
                text_similarity = self.calculate_text_similarity(
                    text1[-100:] if len(text1) > 100 else text1, 
                    text2[:100] if len(text2) > 100 else text2
                )
            
            # If similarity is high, merge clusters
            if text_similarity > 0.3:
                merges.append((meta1['cluster_id'], meta2['cluster_id']))
        
        # Process cluster merges efficiently
        if merges:
            # Build merge mapping
            cluster_mapping = {}
            for c1, c2 in merges:
                # Always map to the smaller cluster ID
                source = max(c1, c2)
                target = min(c1, c2)
                
                # Handle transitive merges
                while target in cluster_mapping:
                    target = cluster_mapping[target]
                
                cluster_mapping[source] = target
            
            # Apply merges in one pass
            if cluster_mapping:
                for i, meta in enumerate(self.page_metadata):
                    cluster_id = meta['cluster_id']
                    if cluster_id in cluster_mapping:
                        # Follow the chain to find the final target
                        target = cluster_id
                        while target in cluster_mapping:
                            target = cluster_mapping[target]
                        self.page_metadata[i]['cluster_id'] = target
    
    def organize_clusters(self):
        """
        Organize pages into document clusters, with language information.
        """
        logger.info("Organizing document clusters...")
        
        # Group pages by cluster ID
        clusters = {}
        for i, meta in enumerate(self.page_metadata):
            cluster_id = meta['cluster_id']
            if cluster_id is not None:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(i)
        
        # Create document cluster objects
        self.document_clusters = []
        for cluster_id, page_indices in clusters.items():
            # Get pages in this cluster
            cluster_pages = [self.page_metadata[i] for i in page_indices]
            
            # Determine document type (majority vote)
            doc_types = [page['doc_type'] for page in cluster_pages if 'doc_type' in page]
            if doc_types:
                doc_type = max(set(doc_types), key=doc_types.count)
            else:
                doc_type = 'unknown'
            
            # Determine language (majority vote)
            languages = [page.get('language', 'unknown') for page in cluster_pages]
            if languages and not all(lang == 'unknown' for lang in languages):
                # Filter out unknown languages for the vote
                known_languages = [lang for lang in languages if lang != 'unknown']
                if known_languages:
                    language = max(set(known_languages), key=known_languages.count)
                else:
                    language = 'unknown'
            else:
                language = 'unknown'
            
            # Create cluster object
            cluster = {
                'cluster_id': cluster_id,
                'page_count': len(page_indices),
                'document_type': doc_type,
                'language': language,
                'pages': sorted(cluster_pages, key=lambda x: (x['pdf_index'], x['page_number'])),
                'pdf_files': list(set(page['pdf_file'] for page in cluster_pages))
            }
            
            self.document_clusters.append(cluster)
        
        # Sort clusters by size (largest first)
        self.document_clusters.sort(key=lambda x: x['page_count'], reverse=True)
        
        logger.info(f"Found {len(self.document_clusters)} document clusters")
    
    def export_results(self):
        """
        Export clustering results, with language information.
        """
        logger.info("Exporting results...")
        
        # Create summary CSV
        cluster_summary = []
        for cluster in self.document_clusters:
            cluster_summary.append({
                'cluster_id': cluster['cluster_id'],
                'document_type': cluster['document_type'],
                'language': cluster.get('language', 'unknown'),
                'page_count': cluster['page_count'],
                'pdf_files': ', '.join(cluster['pdf_files']),
                'first_page': f"{cluster['pages'][0]['pdf_file']} - p{cluster['pages'][0]['page_number']}"
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(cluster_summary)
        summary_path = os.path.join(self.output_dir, "cluster_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Cluster summary saved to {summary_path}")
        
        # Generate language statistics
        language_stats = summary_df['language'].value_counts().reset_index()
        language_stats.columns = ['language', 'cluster_count']
        language_stats_path = os.path.join(self.output_dir, "language_statistics.csv")
        language_stats.to_csv(language_stats_path, index=False)
        logger.info(f"Language statistics saved to {language_stats_path}")
        
        # Save detailed cluster information
        for cluster in self.document_clusters:
            cluster_id = cluster['cluster_id']
            
            # Create detailed page listing
            page_details = []
            for page in cluster['pages']:
                page_details.append({
                    'pdf_file': page['pdf_file'],
                    'page_number': page['page_number'],
                    'doc_type': page.get('doc_type', 'unknown'),
                    'language': page.get('language', 'unknown'),
                    'text_length': page.get('text_length', 0),
                    'enhanced_image_path': page.get('enhanced_image_path', ''),
                    'text_path': page.get('text_path', '')
                })
                
            # Save detailed cluster information
            details_df = pd.DataFrame(page_details)
            details_path = os.path.join(self.output_dir, "document_clusters", f"cluster_{cluster_id}.csv")
            details_df.to_csv(details_path, index=False)
            
        # Return path to summary file
        return summary_path