import os
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
from langdetect import detect
import shutil
import warnings
from tqdm import tqdm  # For progress bars

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
        self.page_images = []
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
            os.path.join(self.output_dir, "document_clusters"),
            os.path.join(self.output_dir, "cache")
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def load_pdf_files(self):
        """Load PDF files from the input directory."""
        self.pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        
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
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache for {pdf_file}: {e}")
        return None

    def save_to_cache(self, pdf_file, results):
        """Save results to cache."""
        if not self.use_cache:
            return
            
        cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{pdf_file}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error saving cache for {pdf_file}: {e}")
    
    def preprocess_image(self, image):
        """
        Optimize image quality for OCR with faster processing.
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image object
        """
        # Convert to grayscale
        gray_image = image.convert('L')
        
        # Use OpenCV for faster processing
        opencv_image = np.array(gray_image)
        
        # Apply basic contrast enhancement
        opencv_image = cv2.convertScaleAbs(opencv_image, alpha=1.5, beta=0)
        
        # Apply mild Gaussian blur to reduce noise (faster than median filter)
        blurred = cv2.GaussianBlur(opencv_image, (3, 3), 0)
        
        # Apply adaptive thresholding (better than simple thresholding)
        threshold_image = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(threshold_image)
        
        return enhanced_image
        
    def extract_text_from_image(self, image, metadata):
        """
        Extract text from an image using OCR, with optimized language detection.
        
        Args:
            image: PIL Image object
            metadata: Dictionary with page metadata
            
        Returns:
            Tuple of (extracted text, updated metadata)
        """
        # Preprocess the image
        enhanced_image = self.preprocess_image(image)
        
        # Save enhanced image
        image_filename = f"{metadata['pdf_index']}_{metadata['page_number']}.png"
        enhanced_image_path = os.path.join(self.output_dir, "enhanced_images", image_filename)
        enhanced_image.save(enhanced_image_path)
        
        # Update metadata
        metadata['enhanced_image_path'] = enhanced_image_path
        
        # Extract text using single OCR pass with multiple languages
        try:
            # Use Tesseract's language detection capabilities
            custom_config = r'--oem 1 --psm 3 -l eng+spa+fra'
            text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Default language
            detected_lang = 'eng'
            
            # Only attempt language detection if there's enough text
            if text and len(text.strip()) > 50:
                try:
                    lang_code = detect(text)
                    # Map langdetect codes to tesseract codes
                    lang_map = {'en': 'eng', 'es': 'spa', 'fr': 'fra'}
                    detected_lang = lang_map.get(lang_code, 'eng')
                except:
                    pass  # Keep default language if detection fails
            
            # Save extracted text
            text_filename = f"{metadata['pdf_index']}_{metadata['page_number']}.txt"
            text_path = os.path.join(self.output_dir, "text_output", text_filename)
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            
            # Update metadata
            metadata['text_path'] = text_path
            metadata['text_length'] = len(text)
            metadata['has_text'] = len(text.strip()) > 0
            metadata['language'] = detected_lang
            
            # Detect document type based on text patterns
            metadata['doc_type'] = self.detect_document_type(text, detected_lang)
            
            return text, metadata
            
        except Exception as e:
            print(f"Error extracting text from image: {e}")
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
        
        # Check for receipt patterns (works across languages)
        if re.search(lang_patterns['receipt'], text_lower) and \
           re.search(r'\d+[\.,]\d{2}', text) and len(text) < 1000:
            return 'receipt'
        
        # Check for letter patterns
        if re.search(lang_patterns['letter'], text_lower):
            return 'letter'
        
        # Check for report patterns
        if re.search(lang_patterns['report'], text_lower) and \
           len(text) > 1000:
            return 'report'
            
        # Check for form patterns
        if re.search(lang_patterns['form'], text_lower):
            return 'form'
            
        # Default
        return 'document'
        
    def process_single_pdf(self, pdf_path, pdf_index, pdf_filename):
        """
        Process a single PDF file and extract text.
        
        Args:
            pdf_path: Path to the PDF file
            pdf_index: Index of the PDF file
            pdf_filename: Name of the PDF file
            
        Returns:
            Tuple of (images, metadata, texts)
        """
        # Check if cached results are available
        cached_results = self.check_cache(pdf_filename)
        if cached_results:
            return cached_results
            
        start_time = time.time()
        result_images = []
        result_metadata = []
        result_texts = []
        
        try:
            # Convert PDF to images
            pages = convert_from_path(pdf_path, dpi=300)
            
            for i, page in enumerate(pages):
                metadata = {
                    'pdf_file': os.path.basename(pdf_path),
                    'pdf_index': pdf_index,
                    'page_number': i + 1,
                    'original_path': pdf_path,
                    'cluster_id': None
                }
                
                text, updated_metadata = self.extract_text_from_image(page, metadata)
                
                result_images.append(page)
                result_metadata.append(updated_metadata)
                result_texts.append(text)
                
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            
        # Calculate processing time
        processing_time = time.time() - start_time
        self.file_times[pdf_filename] = processing_time
        
        # Cache the results
        results = (result_images, result_metadata, result_texts)
        self.save_to_cache(pdf_filename, results)
        
        return results
        
    def process_all_documents(self):
        """Process all PDF files in parallel and extract text."""
        self.load_pdf_files()
        
        all_page_images = []
        all_page_metadata = []
        all_page_texts = []
        
        # Start overall timer
        self.start_time = time.time()
        
        # Process PDFs in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, pdf_file in enumerate(self.pdf_files):
                pdf_path = os.path.join(self.input_dir, pdf_file)
                futures.append(executor.submit(self.process_single_pdf, pdf_path, i, pdf_file))
            
            # Process results as they complete with a progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                images, metadata, texts = future.result()
                all_page_images.extend(images)
                all_page_metadata.extend(metadata)
                all_page_texts.extend(texts)
        
        # Calculate total processing time
        self.total_time = time.time() - self.start_time
        total_time_formatted = str(timedelta(seconds=int(self.total_time)))
        print(f"\nTotal PDF processing time: {total_time_formatted} (HH:MM:SS)")
        
        self.page_images = all_page_images
        self.page_metadata = all_page_metadata
        self.page_texts = all_page_texts
        
        print(f"Processed {len(self.page_images)} pages from {len(self.pdf_files)} PDF files")
        
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
            
        # Use difflib's SequenceMatcher for text similarity
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def cluster_pages_by_content(self):
        """
        Cluster pages based on text content and visual similarity, with language awareness.
        """
        print("Clustering pages by content...")
        
        # Filter pages with text
        text_pages = [(i, text, self.page_metadata[i].get('language', 'eng')) 
                      for i, text in enumerate(self.page_texts) if text.strip()]
        
        if not text_pages:
            print("No text content found in any pages")
            return
            
        page_indices = [i for i, _, _ in text_pages]
        texts = [text for _, text, _ in text_pages]
        languages = [lang for _, _, lang in text_pages]
        
        # Group pages by language
        language_groups = {}
        for i, (page_idx, text, lang) in enumerate(text_pages):
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append((i, page_idx, text))
        
        # Process each language group separately
        all_clusters = np.ones(len(page_indices)) * -1  # Default to -1 (noise)
        next_cluster_id = 0
        
        for lang, lang_pages in language_groups.items():
            print(f"Processing {len(lang_pages)} pages in language: {lang}")
            
            lang_indices = [i for i, _, _ in lang_pages]
            lang_page_indices = [page_idx for _, page_idx, _ in lang_pages]
            lang_texts = [text for _, _, text in lang_pages]
            
            # Select appropriate stop words
            stop_words = 'english'
            if lang == 'spa':
                stop_words = 'spanish'
            elif lang == 'fra':
                stop_words = 'french'
            
            # Create TF-IDF vectors for text clustering (with optimization)
            max_features = min(1000, max(100, len(lang_texts) * 2))
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
            tfidf_matrix = vectorizer.fit_transform(lang_texts)
            
            # Optimize DBSCAN parameters based on data size
            min_samples = min(2, max(2, len(lang_texts) // 20))
            eps = 0.5
            
            # Cluster using DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
            lang_clusters = dbscan.fit_predict(tfidf_matrix)
            
            # Map cluster IDs to global cluster IDs
            for i, cluster_id in enumerate(lang_clusters):
                if cluster_id >= 0:  # Not noise
                    all_clusters[lang_indices[i]] = cluster_id + next_cluster_id
            
            # Update next_cluster_id
            if lang_clusters.max() >= 0:
                next_cluster_id += lang_clusters.max() + 1
        
        # Update metadata with cluster IDs
        for idx, cluster_id in enumerate(all_clusters):
            page_idx = page_indices[idx]
            
            # Assign cluster ID (DBSCAN uses -1 for noise/outliers)
            self.page_metadata[page_idx]['cluster_id'] = int(cluster_id) if cluster_id >= 0 else None
        
        # Handle pages with no text
        self.process_pages_without_text()
        
        # Check for document continuity (language-aware)
        self.check_document_continuity()
        
        # Organize results
        self.organize_clusters()
        
    def process_pages_without_text(self):
        """
        Process pages that have no text or were not clustered.
        """
        print("Processing pages without text...")
        
        # Group unclustered pages
        unclustered_indices = [i for i, meta in enumerate(self.page_metadata) 
                              if meta['cluster_id'] is None]
        
        # Look for the next available cluster ID
        next_cluster_id = max([meta['cluster_id'] for meta in self.page_metadata 
                              if meta['cluster_id'] is not None], default=-1) + 1
        
        for idx in unclustered_indices:
            self.page_metadata[idx]['cluster_id'] = next_cluster_id
            next_cluster_id += 1
    
    def check_document_continuity(self):
        """
        Check for document continuity based on page numbers and text flow.
        Merge clusters that are likely part of the same document.
        Language-aware to avoid merging documents in different languages.
        """
        print("Checking document continuity...")
        
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
                
            # Check text continuity
            text1 = self.page_texts[idx1]
            text2 = self.page_texts[idx2]
            
            # Calculate text continuity (optimized for speed)
            text_end = text1[-100:] if len(text1) > 100 else text1
            text_start = text2[:100] if len(text2) > 100 else text2
            text_similarity = self.calculate_text_similarity(text_end, text_start)
            
            # If similarity is high, merge clusters
            if text_similarity > 0.3:
                merges.append((meta1['cluster_id'], meta2['cluster_id']))
        
        # Process cluster merges
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
            
            # Apply merges
            for i, meta in enumerate(self.page_metadata):
                if meta['cluster_id'] in cluster_mapping:
                    self.page_metadata[i]['cluster_id'] = cluster_mapping[meta['cluster_id']]
    
    def organize_clusters(self):
        """
        Organize pages into document clusters, with language information.
        """
        print("Organizing document clusters...")
        
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
        
        print(f"Found {len(self.document_clusters)} document clusters")
    
    def export_results(self):
        """
        Export clustering results, with language information.
        """
        print("Exporting results...")
        
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
        
        # Generate language statistics
        language_stats = summary_df['language'].value_counts().reset_index()
        language_stats.columns = ['language', 'cluster_count']
        language_stats_path = os.path.join(self.output_dir, "language_statistics.csv")
        language_stats.to_csv(language_stats_path, index=False)
        
        # Save detailed cluster information
        cluster_details_path = os.path.join(self.output_dir, "document_clusters", "cluster_details.csv")
        page_details = []
        
        for cluster in self.document_clusters:
            cluster_id = cluster['cluster_id']
            for page in cluster['pages']:
                page_details.append({
                    'cluster_id': cluster_id,
                    'pdf_file': page['pdf_file'],
                    'page_number': page['page_number'],
                    'doc_type': page.get('doc_type', 'unknown'),
                    'language': page.get('language', 'unknown'),
                    'text_length': page.get('text_length', 0),
                    'enhanced_image_path': page.get('enhanced_image_path', ''),
                    'text_path': page.get('text_path', '')
                })
        
        # Save detailed page information
        detail_df = pd.DataFrame(page_details)
        detail_df.to_csv(cluster_details_path, index=False)
        
        return summary_path