import os
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import difflib
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import argparse
import time
from datetime import timedelta

class DocumentClusteringPrototype:
    def __init__(self, input_dir, output_dir, sample_size=None):
        """
        Initialize the document clustering prototype.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save results
            sample_size: Number of PDFs to process (None for all)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.pdf_files = []
        self.page_images = []
        self.page_texts = []
        self.page_metadata = []
        self.document_clusters = []
        
        # Timing information
        self.start_time = None
        self.file_times = {}
        self.total_time = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "enhanced_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "text_output"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "document_clusters"), exist_ok=True)
        
    def load_pdf_files(self):
        """Load PDF files from the input directory."""
        self.pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        
        if self.sample_size:
            self.pdf_files = self.pdf_files[:min(self.sample_size, len(self.pdf_files))]
            
        print(f"Found {len(self.pdf_files)} PDF files to process")
        
    def convert_pdfs_to_images(self, pdf_path, pdf_index):
        """
        Convert a PDF file to images.
        
        Args:
            pdf_path: Path to the PDF file
            pdf_index: Index of the PDF in the list of files
            
        Returns:
            List of (image, metadata) tuples
        """
        print(f"Converting PDF {pdf_index+1}/{len(self.pdf_files)}: {pdf_path}")
        
        result = []
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
                result.append((page, metadata))
                
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            
        return result
    
    def preprocess_image(self, image):
        """
        Enhance image quality for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image object
        """
        # Convert to grayscale
        gray_image = image.convert('L')
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(gray_image)
        contrast_image = enhancer.enhance(2.0)
        
        # Apply sharpening
        sharp_image = contrast_image.filter(ImageFilter.SHARPEN)
        
        # Denoise using median filter
        opencv_image = np.array(sharp_image)
        denoised_image = cv2.medianBlur(opencv_image, 3)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(denoised_image)
        
        # Binarize (adaptive thresholding would be better but this is simpler)
        threshold_image = enhanced_image.point(lambda p: p > 128 and 255)
        
        return threshold_image
    
    def extract_text_from_image(self, image, metadata):
        """
        Extract text from an image using OCR, with language detection.
        
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
        
        # Extract text using Tesseract OCR with multiple language support
        try:
            # First try with English OCR
            custom_config = r'--oem 1 --psm 3 -l eng'
            eng_text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Try with Spanish
            custom_config = r'--oem 1 --psm 3 -l spa'
            spa_text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Try with French
            custom_config = r'--oem 1 --psm 3 -l fra'
            fra_text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Determine the most likely language by comparing text length and common words
            lang_scores = self.score_language_match(eng_text, spa_text, fra_text)
            
            # Get the language with the highest score
            best_lang = max(lang_scores.items(), key=lambda x: x[1])
            detected_lang = best_lang[0]
            
            # Select the text from the best language
            if detected_lang == 'eng':
                text = eng_text
            elif detected_lang == 'spa':
                text = spa_text
            else:  # fra
                text = fra_text
            
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
            
            # Detect document type based on text patterns (language-aware)
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
            
    def score_language_match(self, eng_text, spa_text, fra_text):
        """
        Score how well the extracted text matches each language.
        
        Args:
            eng_text: Text extracted using English OCR
            spa_text: Text extracted using Spanish OCR
            fra_text: Text extracted using French OCR
            
        Returns:
            Dictionary with language codes as keys and scores as values
        """
        # Initialize scores
        scores = {'eng': 0, 'spa': 0, 'fra': 0}
        
        # Score based on text length (longer text often indicates better language match)
        scores['eng'] += len(eng_text.strip()) * 0.01
        scores['spa'] += len(spa_text.strip()) * 0.01
        scores['fra'] += len(fra_text.strip()) * 0.01
        
        # Common English words
        eng_words = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but']
        # Common Spanish words
        spa_words = ['el', 'la', 'que', 'de', 'en', 'y', 'un', 'ser', 'por', 'con']
        # Common French words
        fra_words = ['le', 'la', 'les', 'un', 'une', 'et', 'de', 'que', 'pas', 'pour']
        
        # Check for presence of common words in each language
        eng_text_lower = eng_text.lower()
        for word in eng_words:
            if f" {word} " in eng_text_lower or eng_text_lower.startswith(f"{word} ") or eng_text_lower.endswith(f" {word}"):
                scores['eng'] += 10
                
        spa_text_lower = spa_text.lower()
        for word in spa_words:
            if f" {word} " in spa_text_lower or spa_text_lower.startswith(f"{word} ") or spa_text_lower.endswith(f" {word}"):
                scores['spa'] += 10
                
        fra_text_lower = fra_text.lower()
        for word in fra_words:
            if f" {word} " in fra_text_lower or fra_text_lower.startswith(f"{word} ") or fra_text_lower.endswith(f" {word}"):
                scores['fra'] += 10
        
        return scores
    
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
    
    def process_all_documents(self):
        """Process all PDF files and extract text."""
        self.load_pdf_files()
        
        all_page_images = []
        all_page_metadata = []
        
        # Start overall timer
        self.start_time = time.time()
        
        # Process each PDF file
        for i, pdf_file in enumerate(self.pdf_files):
            # Start timer for this file
            file_start_time = time.time()
            
            pdf_path = os.path.join(self.input_dir, pdf_file)
            print(f"Processing {i+1}/{len(self.pdf_files)}: {pdf_file}")
            
            page_results = self.convert_pdfs_to_images(pdf_path, i)
            
            # Process each page
            for image, metadata in page_results:
                text, updated_metadata = self.extract_text_from_image(image, metadata)
                
                all_page_images.append(image)
                all_page_metadata.append(updated_metadata)
                self.page_texts.append(text)
            
            # End timer for this file and store the duration
            file_end_time = time.time()
            duration = file_end_time - file_start_time
            self.file_times[pdf_file] = duration
            
            # Format and print the time for this file
            duration_formatted = str(timedelta(seconds=int(duration)))
            print(f"Completed {pdf_file} in {duration_formatted} (HH:MM:SS)")
        
        # Calculate total processing time
        self.total_time = time.time() - self.start_time
        total_time_formatted = str(timedelta(seconds=int(self.total_time)))
        print(f"\nTotal processing time: {total_time_formatted} (HH:MM:SS)")
        
        self.page_images = all_page_images
        self.page_metadata = all_page_metadata
        
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
            
            # Create TF-IDF vectors for text clustering
            vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words)
            tfidf_matrix = vectorizer.fit_transform(lang_texts)
            
            # Cluster using DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
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
        
        # Assign individual cluster IDs to "orphan" pages
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
            
            # Calculate text continuity
            text_similarity = self.calculate_text_similarity(text1[-100:] if len(text1) > 100 else text1, 
                                                          text2[:100] if len(text2) > 100 else text2)
            
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

def main():
    parser = argparse.ArgumentParser(description='Document Clustering Prototype')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    parser.add_argument('--sample', '-s', type=int, help='Number of PDFs to process (default: all)')
    
    args = parser.parse_args()
    
    # Create and run the prototype
    prototype = DocumentClusteringPrototype(args.input, args.output, args.sample)
    prototype.process_all_documents()
    prototype.cluster_pages_by_content()
    summary_path = prototype.export_results()
    
    print(f"Processing complete. Summary available at {summary_path}")

if __name__ == "__main__":
    main()