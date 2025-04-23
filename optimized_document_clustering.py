# filename: optimized_document_clustering_modified.py
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
        Extract text from an image using OCR, with optimized language detection
        and potentially improved handwriting recognition settings.

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
            # --- MODIFIED TESSERACT CONFIGURATION ---
            # Use Tesseract's language detection capabilities
            # Changed --psm from 3 to 6 (Assume a single uniform block of text)
            # Added script/Latin for potentially better handwriting recognition
            # Ensure 'script/Latin' traineddata is installed for Tesseract
            custom_config = r'--oem 1 --psm 6 -l eng+spa+fra+script/Latin'
            # --- END MODIFICATION ---

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
            print(f"Error extracting text from image {metadata.get('pdf_file', '')} page {metadata.get('page_number', '')}: {e}")
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
                'receipt': r'receipt|total|subtotal|payment|paid|amount|\<span class="math-inline">\|€\|£',
                    'letter'\: r'dear\\s\+\[a\-z,\\\.\]\+\|sincerely\|regards\|yours truly',
                'report'\: r'report\|analysis\|summary\|findings\|conclusion\|introduction',
                'form'\: r'form\|please fill\|checkbox\|check one\|signature\|date of birth'
                \},
'spa'\: \{
'receipt'\: r'recibo\|total\|subtotal\|pago\|pagado\|importe\|\\$\|€\|£',
'letter'\: r'estimado\|querido\|atentamente\|saludos\|cordialmente',
'report'\: r'informe\|análisis\|resumen\|hallazgos\|conclusión\|introducción',
'form'\: r'formulario\|por favor\|casilla\|marque\|firma\|fecha de nacimiento'
\},
'fra'\: \{
'receipt'\: r'reçu\|total\|sous\-total\|paiement\|payé\|montant\|\\</span>|€|£',
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
            # Make sure cached results have all expected keys, add defaults if missing
            images, metadata_list, texts = cached_results
            for meta in metadata_list:
                meta.setdefault('pdf_file', pdf_filename)
                meta.setdefault('pdf_index', pdf_index)
                meta.setdefault('original_path', pdf_path)
                meta.setdefault('cluster_id', None)
                meta.setdefault('enhanced_image_path', None)
                meta.setdefault('text_path', None)
                meta.setdefault('text_length', 0)
                meta.setdefault('has_text', False)
                meta.setdefault('language', 'unknown')
                meta.setdefault('doc_type', 'unknown')
            return images, metadata_list, texts

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

                # Don't store full PIL images in memory long term if not needed,
                # but we need them for the initial processing step.
                # Consider if result_images is truly needed downstream or if paths suffice.
                # For now, keep original behavior.
                result_images.append(page)
                result_metadata.append(updated_metadata)
                result_texts.append(text)

        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

        # Calculate processing time
        processing_time = time.time() - start_time
        self.file_times[pdf_filename] = processing_time

        # Cache the results
        # Avoid caching large image objects if possible.
        # Store paths instead if the images themselves aren't needed later.
        # Here, caching (metadata, texts) might be better.
        # Keeping original behavior for now.
        results_to_cache = (result_images, result_metadata, result_texts)
        self.save_to_cache(pdf_filename, results_to_cache)

        return results_to_cache

    def process_all_documents(self):
        """Process all PDF files in parallel and extract text."""
        self.load_pdf_files()

        all_page_images = [] # Consider removing if images aren't needed later
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
                try:
                    images, metadata, texts = future.result()
                    # all_page_images.extend(images) # Append only if needed
                    all_page_metadata.extend(metadata)
                    all_page_texts.extend(texts)
                except Exception as e:
                    print(f"Error processing future result: {e}")


        # Calculate total processing time
        self.total_time = time.time() - self.start_time
        total_time_formatted = str(timedelta(seconds=int(self.total_time)))
        print(f"\nTotal PDF processing time: {total_time_formatted} (HH:MM:SS)")

        # self.page_images = all_page_images # Assign only if needed
        self.page_metadata = all_page_metadata
        self.page_texts = all_page_texts

        # Use len(self.page_metadata) as the count of processed pages
        print(f"Processed {len(self.page_metadata)} pages from {len(self.pdf_files)} PDF files")


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
                      for i, text in enumerate(self.page_texts) if self.page_metadata[i].get('has_text', False)]


        if not text_pages:
            print("No text content found in any pages to cluster.")
            # Assign all pages to a single 'unclustered' group or handle as needed
            next_cluster_id = 0
            for i in range(len(self.page_metadata)):
                 self.page_metadata[i]['cluster_id'] = next_cluster_id
            self.organize_clusters() # Organize even if unclustered
            return


        page_indices = [i for i, _, _ in text_pages]
        texts = [text for _, text, _ in text_pages]
        languages = [lang for _, _, lang in text_pages]

        # Group pages by language
        language_groups = {}
        for i, (page_idx, text, lang) in enumerate(text_pages):
            if lang not in language_groups:
                language_groups[lang] = []
            # Store the index within the text_pages list, the original page index, and the text
            language_groups[lang].append((i, page_idx, text))


        all_clusters = np.ones(len(page_indices), dtype=int) * -1  # Default to -1 (noise/unclustered)
        next_cluster_id = 0

        for lang, lang_pages in language_groups.items():
            print(f"Processing {len(lang_pages)} pages in language: {lang}")

            if len(lang_pages) < 2: # Need at least 2 samples for DBSCAN
                print(f"Skipping clustering for language '{lang}' due to insufficient pages ({len(lang_pages)}).")
                continue

            # Get indices relative to the 'text_pages' list
            lang_indices_in_text_pages = [i for i, _, _ in lang_pages]
            # Get original page indices
            lang_page_indices = [page_idx for _, page_idx, _ in lang_pages]
            lang_texts = [text for _, _, text in lang_pages]


            # Select appropriate stop words
            stop_words = 'english'
            if lang == 'spa':
                stop_words = 'spanish' # Ensure scikit-learn supports these language names
            elif lang == 'fra':
                stop_words = 'french'

            try:
                 # Create TF-IDF vectors for text clustering (with optimization)
                 max_features = min(1000, max(100, len(lang_texts) * 2))
                 vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
                 tfidf_matrix = vectorizer.fit_transform(lang_texts)

                 # Optimize DBSCAN parameters based on data size
                 min_samples = min(max(2, len(lang_texts) // 20), 5) # Clamp min_samples between 2 and 5
                 eps = 0.6 # Adjusted epsilon slightly, may need tuning

                 # Cluster using DBSCAN
                 dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
                 lang_clusters = dbscan.fit_predict(tfidf_matrix)

                 # Map cluster IDs to global cluster IDs
                 unique_lang_clusters = set(lang_clusters)
                 cluster_id_map = {}
                 for cluster_id in unique_lang_clusters:
                     if cluster_id >= 0: # Not noise
                         cluster_id_map[cluster_id] = next_cluster_id
                         next_cluster_id += 1

                 # Assign global cluster IDs back to the all_clusters array
                 for i, cluster_id in enumerate(lang_clusters):
                     if cluster_id >= 0:
                         # Use the index relative to text_pages list
                         idx_in_all_clusters = lang_indices_in_text_pages[i]
                         all_clusters[idx_in_all_clusters] = cluster_id_map[cluster_id]

            except ValueError as ve:
                 print(f"Skipping clustering for language '{lang}': {ve}")
                 # Assign these pages to a single 'unclustered' group for this language
                 lang_unclustered_id = next_cluster_id
                 next_cluster_id += 1
                 for i in range(len(lang_indices_in_text_pages)):
                    idx_in_all_clusters = lang_indices_in_text_pages[i]
                    all_clusters[idx_in_all_clusters] = lang_unclustered_id
            except Exception as e:
                 print(f"Error during clustering for language '{lang}': {e}")


        # --- Update metadata with cluster IDs ---
        # Map the results from 'all_clusters' (which corresponds to 'text_pages')
        # back to the original 'self.page_metadata' list.
        clustered_page_indices_set = set()
        for idx_in_text_pages, cluster_id in enumerate(all_clusters):
            if cluster_id >= 0: # Successfully clustered (not noise)
                original_page_idx = page_indices[idx_in_text_pages]
                self.page_metadata[original_page_idx]['cluster_id'] = int(cluster_id)
                clustered_page_indices_set.add(original_page_idx)

        # Handle pages that were filtered out initially (no text) or marked as noise by DBSCAN (-1)
        noise_or_no_text_id = next_cluster_id
        assigned_noise_cluster = False
        for i in range(len(self.page_metadata)):
            if i not in clustered_page_indices_set:
                # Check if it was DBSCAN noise or had no text initially
                meta = self.page_metadata[i]
                is_noise = False
                if meta.get('has_text', False):
                    try:
                        # Find its index in text_pages if it had text
                        text_page_idx = page_indices.index(i)
                        if all_clusters[text_page_idx] == -1:
                             is_noise = True
                    except ValueError:
                        pass # Should not happen if has_text is True, but safety check

                if is_noise or not meta.get('has_text', False):
                    self.page_metadata[i]['cluster_id'] = noise_or_no_text_id
                    assigned_noise_cluster = True

        if assigned_noise_cluster:
             next_cluster_id += 1 # Increment if the noise cluster ID was used

        # Check for document continuity (language-aware)
        self.check_document_continuity()

        # Organize results
        self.organize_clusters()


    def process_pages_without_text(self):
        """
        Assigns a unique cluster ID to pages without text or those failing initial clustering.
        NOTE: This logic is now integrated into the end of cluster_pages_by_content.
              This function can be potentially removed if redundancy is confirmed.
        """
        print("Processing pages without text or marked as noise...")

        # Find the next available cluster ID after clustering
        max_cluster_id = -1
        for meta in self.page_metadata:
            if meta.get('cluster_id') is not None:
                 max_cluster_id = max(max_cluster_id, meta['cluster_id'])
        next_cluster_id = max_cluster_id + 1

        assigned_new_cluster = False
        for i, meta in enumerate(self.page_metadata):
            # Assign a cluster ID if it's currently None (e.g., no text pages handled separately)
            # or potentially if it was marked as noise (-1) depending on desired handling.
            # The logic in cluster_pages_by_content now handles this.
            if meta.get('cluster_id') is None: # Check specifically for None
                self.page_metadata[i]['cluster_id'] = next_cluster_id
                assigned_new_cluster = True

        if assigned_new_cluster:
            print(f"Assigned cluster ID {next_cluster_id} to pages initially without text/clusters.")
            # Increment for potential future use, though likely not needed here
            # next_cluster_id += 1


    def check_document_continuity(self):
        """
        Check for document continuity based on page numbers and text flow.
        Merge clusters that are likely part of the same document.
        Language-aware to avoid merging documents in different languages.
        """
        print("Checking document continuity...")

        # Sort pages by PDF and page number
        sorted_page_indices = sorted(range(len(self.page_metadata)),
                                     key=lambda k: (self.page_metadata[k]['pdf_index'], self.page_metadata[k]['page_number']))

        # Group potential merges by the target cluster ID
        potential_merges = defaultdict(set) # target_cluster_id -> {source_cluster_id1, source_cluster_id2, ...}

        for i in range(len(sorted_page_indices) - 1):
            idx1 = sorted_page_indices[i]
            idx2 = sorted_page_indices[i+1]

            meta1 = self.page_metadata[idx1]
            meta2 = self.page_metadata[idx2]

            # Only check pages in the same PDF
            if meta1['pdf_index'] != meta2['pdf_index']:
                continue

            # Only check if pages are consecutive
            if meta2['page_number'] != meta1['page_number'] + 1:
                continue

            # Skip if either page doesn't have a valid cluster ID assigned yet
            c1 = meta1.get('cluster_id')
            c2 = meta2.get('cluster_id')
            if c1 is None or c2 is None:
                continue

            # Skip if pages are already in the same cluster
            if c1 == c2:
                continue

            # Check if pages are in the same language (allow unknown matches)
            lang1 = meta1.get('language', 'unknown')
            lang2 = meta2.get('language', 'unknown')

            # If languages are different and both are known, don't merge
            if lang1 != 'unknown' and lang2 != 'unknown' and lang1 != lang2:
                continue

            # Check text continuity only if both pages have text
            text1 = self.page_texts[idx1] if meta1.get('has_text') else ""
            text2 = self.page_texts[idx2] if meta2.get('has_text') else ""

            if text1 and text2:
                # Calculate text continuity (optimized for speed)
                text_end = text1[-100:] # No need for length check, slicing handles it
                text_start = text2[:100]
                text_similarity = self.calculate_text_similarity(text_end, text_start)

                # If similarity is high enough, consider merging
                # Threshold might need adjustment based on document types
                if text_similarity > 0.25:
                    target_cluster = min(c1, c2)
                    source_cluster = max(c1, c2)
                    potential_merges[target_cluster].add(source_cluster)
            # Optional: Add logic here to potentially merge based purely on sequence
            # if one or both pages lack text, but be cautious.
            # else:
            #    # Merge based on sequence alone? Risky.
            #    target_cluster = min(c1, c2)
            #    source_cluster = max(c1, c2)
            #    potential_merges[target_cluster].add(source_cluster)


        # Process cluster merges iteratively to handle chains
        if potential_merges:
            print(f"Found {len(potential_merges)} potential cluster merge targets.")
            final_mapping = {} # source_id -> final_target_id
            processed_targets = set()

            # Find the ultimate target for each cluster involved in merges
            all_involved_clusters = set(potential_merges.keys()) | set.union(*potential_merges.values())

            for cluster_id in sorted(list(all_involved_clusters)): # Process in order
                if cluster_id in final_mapping: # Already remapped
                    continue

                current_target = cluster_id
                visited = {current_target}
                source_clusters_to_remap = {current_target}

                # Follow the merge chain upwards (source -> target)
                while True:
                    found_new_target = False
                    # Check if any cluster wants to merge *into* current_target
                    merged_into_current = False
                    for target, sources in potential_merges.items():
                         if current_target in sources:
                              # Check if this target is smaller or already processed differently
                              if target < current_target and target not in visited:
                                   current_target = target
                                   visited.add(current_target)
                                   found_new_target = True
                                   merged_into_current = True
                                   break # Restart check with new target
                              # else: might be a loop or higher target, ignore for now

                    # Also add all clusters that current_target wants to merge
                    if current_target in potential_merges:
                        sources_to_add = potential_merges[current_target] - visited
                        if sources_to_add:
                             source_clusters_to_remap.update(sources_to_add)
                             visited.update(sources_to_add)
                             # This doesn't change the target, just the set to remap

                    if not found_new_target: # Reached the root target for this chain
                         break


                # Map all involved clusters in this chain to the final root target
                for source_id in source_clusters_to_remap:
                     if source_id != current_target: # Don't map target to itself
                          final_mapping[source_id] = current_target


            # Apply merges if any mapping was created
            if final_mapping:
                 print(f"Remapping {len(final_mapping)} source clusters.")
                 for i, meta in enumerate(self.page_metadata):
                     original_cluster_id = meta.get('cluster_id')
                     if original_cluster_id is not None and original_cluster_id in final_mapping:
                          self.page_metadata[i]['cluster_id'] = final_mapping[original_cluster_id]
                 print("Document continuity merging complete.")


    def organize_clusters(self):
        """
        Organize pages into document clusters, with language information.
        """
        print("Organizing document clusters...")

        # Group pages by final cluster ID
        clusters = defaultdict(list)
        for i, meta in enumerate(self.page_metadata):
            cluster_id = meta.get('cluster_id')
            # Treat None or negative IDs carefully if they exist after continuity check
            if cluster_id is not None and cluster_id >= 0:
                clusters[cluster_id].append(i)
            elif cluster_id is not None: # Handle potential noise/unclustered IDs explicitly
                 # Maybe group all negative IDs into one "noise" cluster?
                 clusters['noise'].append(i)


        # Create document cluster objects
        self.document_clusters = []
        cluster_ids_processed = set()

        sorted_cluster_ids = sorted([cid for cid in clusters.keys() if isinstance(cid, int)])
        if 'noise' in clusters:
             sorted_cluster_ids.append('noise') # Process noise last


        for cluster_id in sorted_cluster_ids:
            if cluster_id in cluster_ids_processed: continue

            page_indices = clusters[cluster_id]
            cluster_ids_processed.add(cluster_id)

            if not page_indices: continue # Should not happen with defaultdict

            # Get pages in this cluster
            cluster_pages_meta = [self.page_metadata[i] for i in page_indices]

            # Sort pages within the cluster
            cluster_pages_meta.sort(key=lambda x: (x['pdf_index'], x['page_number']))


            # Determine document type (majority vote, fallback to 'document' or 'unknown')
            doc_types = [page.get('doc_type', 'unknown') for page in cluster_pages_meta]
            if doc_types:
                 known_doc_types = [dt for dt in doc_types if dt != 'unknown']
                 if known_doc_types:
                      doc_type = max(set(known_doc_types), key=known_doc_types.count)
                 else:
                      # If all were unknown, check if it's the noise cluster
                      doc_type = 'noise_cluster' if cluster_id == 'noise' else 'unknown'
            else:
                 doc_type = 'noise_cluster' if cluster_id == 'noise' else 'unknown'


            # Determine language (majority vote, fallback to 'unknown')
            languages = [page.get('language', 'unknown') for page in cluster_pages_meta]
            if languages:
                known_languages = [lang for lang in languages if lang != 'unknown']
                if known_languages:
                    language = max(set(known_languages), key=known_languages.count)
                else:
                    language = 'unknown'
            else:
                language = 'unknown'

            # Create cluster object
            cluster = {
                'cluster_id': cluster_id, # Can be int or 'noise'
                'page_count': len(page_indices),
                'document_type': doc_type,
                'language': language,
                'pages': cluster_pages_meta, # List of metadata dicts
                'pdf_files': sorted(list(set(page['pdf_file'] for page in cluster_pages_meta)))
            }

            self.document_clusters.append(cluster)


        # Sort clusters by ID for consistent output (optional: sort by size)
        self.document_clusters.sort(key=lambda x: x['cluster_id'] if isinstance(x['cluster_id'], int) else float('inf')) # Put 'noise' last

        print(f"Organized into {len(self.document_clusters)} final document clusters (including potential noise cluster).")


    def export_results(self):
        """
        Export clustering results, with language information.
        """
        print("Exporting results...")

        # Create summary CSV
        cluster_summary = []
        for cluster in self.document_clusters:
            cluster_id = cluster['cluster_id']
            first_page_meta = cluster['pages'][0] if cluster['pages'] else {}
            cluster_summary.append({
                'cluster_id': cluster_id,
                'document_type': cluster['document_type'],
                'language': cluster.get('language', 'unknown'),
                'page_count': cluster['page_count'],
                'pdf_files': ', '.join(cluster['pdf_files']),
                'first_page': f"{first_page_meta.get('pdf_file','NA')} - p{first_page_meta.get('page_number','NA')}"
            })

        # Save summary to CSV
        summary_df = pd.DataFrame(cluster_summary)
        summary_path = os.path.join(self.output_dir, "cluster_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Cluster summary saved to: {summary_path}")

        # Generate language statistics from summary
        if not summary_df.empty:
             language_stats = summary_df['language'].value_counts().reset_index()
             language_stats.columns = ['language', 'cluster_count']
             language_stats_path = os.path.join(self.output_dir, "language_statistics.csv")
             language_stats.to_csv(language_stats_path, index=False)
             print(f"Language statistics saved to: {language_stats_path}")
        else:
             print("Skipping language statistics (no clusters found).")


        # Save detailed cluster information (all pages)
        cluster_details_path = os.path.join(self.output_dir, "cluster_details.csv") # Changed dir
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
                    'has_text': page.get('has_text', False),
                    'enhanced_image_path': page.get('enhanced_image_path', ''),
                    'text_path': page.get('text_path', '')
                })

        # Save detailed page information
        if page_details:
             detail_df = pd.DataFrame(page_details)
             # Ensure output directory exists before saving
             os.makedirs(os.path.dirname(cluster_details_path), exist_ok=True)
             detail_df.to_csv(cluster_details_path, index=False)
             print(f"Detailed page information saved to: {cluster_details_path}")
        else:
             print("Skipping detailed page information (no pages processed).")


        # --- Create links to text files organized by cluster ---
        print("Organizing text files by cluster...")
        cluster_text_dir = os.path.join(self.output_dir, "document_clusters")
        # Clear old links/files in this specific dir if needed, be careful
        # if os.path.exists(cluster_text_dir):
        #      shutil.rmtree(cluster_text_dir) # Use with caution!
        os.makedirs(cluster_text_dir, exist_ok=True)

        for cluster in self.document_clusters:
             cluster_id_str = str(cluster['cluster_id'])
             target_cluster_dir = os.path.join(cluster_text_dir, f"cluster_{cluster_id_str}")
             os.makedirs(target_cluster_dir, exist_ok=True)

             for page_meta in cluster['pages']:
                  text_path = page_meta.get('text_path')
                  if text_path and os.path.exists(text_path):
                       # Create a meaningful link name
                       link_name = f"{page_meta['pdf_file']}_p{page_meta['page_number']}.txt"
                       link_path = os.path.join(target_cluster_dir, link_name)
                       try:
                            # Use relative path for source if possible, makes it more portable
                            rel_text_path = os.path.relpath(text_path, start=target_cluster_dir)
                            if os.path.exists(link_path):
                                 os.remove(link_path) # Remove existing link first
                            os.symlink(rel_text_path, link_path)
                       except Exception as e:
                            print(f"Warning: Could not create symlink for {text_path} in cluster {cluster_id_str}: {e}")
                  # Optionally link enhanced images too
                  # image_path = page_meta.get('enhanced_image_path')
                  # ... similar linking logic ...

        print("Text file organization complete.")
        return summary_path


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Document Clustering")
    parser.add_argument("--input", required=True, help="Input directory containing PDF files")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--sample", type=int, default=None, help="Number of PDFs to process (default: all)")
    parser.add_argument("--workers", type=int, default=None, help="Maximum number of worker processes (default: CPU count - 1)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of results")
    parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache before processing")

    args = parser.parse_args()

    if args.clear_cache:
        cache_dir = os.path.join(args.output, "cache")
        if os.path.exists(cache_dir):
            print(f"Clearing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir) # Recreate empty cache dir

    # Instantiate and run the clustering process
    clustering = OptimizedDocumentClustering(
        input_dir=args.input,
        output_dir=args.output,
        sample_size=args.sample,
        max_workers=args.workers,
        use_cache=not args.no_cache
    )

    clustering.process_all_documents()
    if clustering.page_metadata: # Only cluster if pages were processed
         clustering.cluster_pages_by_content()
         # Continuity check and organizing happens within cluster_pages_by_content now
         clustering.export_results()
    else:
         print("No pages were processed. Skipping clustering and export.")


    print("Document clustering process finished.")