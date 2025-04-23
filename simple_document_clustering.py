"""
Simplified Document Clustering Script

This is a simplified version of the document clustering script that focuses on the core functionality
without the complexity of the full implementation. It helps diagnose issues with the original script.
"""

import os
import logging
import argparse
import time
from datetime import timedelta
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDocumentClustering:
    def __init__(self, input_dir, output_dir, sample_size=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.pdf_files = []
        self.page_texts = []
        self.page_metadata = []
        self.clusters = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def process_pdfs(self):
        """Process PDFs and extract text"""
        # Get list of PDF files
        self.pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        
        if self.sample_size:
            self.pdf_files = self.pdf_files[:min(self.sample_size, len(self.pdf_files))]
            
        logger.info(f"Processing {len(self.pdf_files)} PDF files")
        
        # Process each PDF
        for pdf_index, pdf_file in enumerate(self.pdf_files):
            pdf_path = os.path.join(self.input_dir, pdf_file)
            logger.info(f"Processing {pdf_index+1}/{len(self.pdf_files)}: {pdf_file}")
            
            try:
                # Convert PDF to images
                pages = convert_from_path(pdf_path, dpi=300)
                
                # Process each page
                for page_num, page_image in enumerate(pages):
                    # Extract text
                    text = pytesseract.image_to_string(page_image)
                    
                    # Store metadata
                    metadata = {
                        'pdf_file': pdf_file,
                        'pdf_index': pdf_index,
                        'page_number': page_num + 1,
                        'text_length': len(text),
                        'cluster_id': None
                    }
                    
                    self.page_texts.append(text)
                    self.page_metadata.append(metadata)
                    
                    logger.info(f"  Processed page {page_num+1}, extracted {len(text)} characters")
            
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
        
        logger.info(f"Processed {len(self.page_texts)} pages from {len(self.pdf_files)} PDF files")
        
    def cluster_documents(self):
        """Cluster documents based on text content"""
        if not self.page_texts:
            logger.warning("No text to cluster")
            return
            
        logger.info("Clustering documents...")
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.page_texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Cluster using DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
        clusters = dbscan.fit_predict(distance_matrix)
        
        # Update metadata with cluster IDs
        for i, cluster_id in enumerate(clusters):
            if i < len(self.page_metadata):
                self.page_metadata[i]['cluster_id'] = int(cluster_id) if cluster_id >= 0 else None
        
        # Group pages by cluster
        cluster_groups = {}
        for i, meta in enumerate(self.page_metadata):
            cluster_id = meta.get('cluster_id')
            if cluster_id is not None and cluster_id >= 0:
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(i)
        
        # Create cluster objects
        self.clusters = []
        for cluster_id, page_indices in cluster_groups.items():
            cluster_pages = [self.page_metadata[i] for i in page_indices]
            cluster = {
                'cluster_id': cluster_id,
                'page_count': len(page_indices),
                'pages': cluster_pages,
                'pdf_files': list(set(page['pdf_file'] for page in cluster_pages))
            }
            self.clusters.append(cluster)
        
        logger.info(f"Found {len(self.clusters)} clusters")
        
    def export_results(self):
        """Export clustering results"""
        logger.info("Exporting results...")
        
        # Create summary CSV
        if not self.clusters:
            logger.warning("No clusters to export")
            # Create an empty summary file
            with open(os.path.join(self.output_dir, "cluster_summary.txt"), "w") as f:
                f.write("No clusters found.\n")
                f.write(f"Processed {len(self.page_texts)} pages from {len(self.pdf_files)} PDF files.\n")
            return os.path.join(self.output_dir, "cluster_summary.txt")
            
        # Create cluster summary
        cluster_summary = []
        for cluster in self.clusters:
            cluster_summary.append({
                'cluster_id': cluster['cluster_id'],
                'page_count': cluster['page_count'],
                'pdf_files': ', '.join(cluster['pdf_files']),
                'first_page': f"{cluster['pages'][0]['pdf_file']} - p{cluster['pages'][0]['page_number']}"
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(cluster_summary)
        summary_path = os.path.join(self.output_dir, "cluster_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Cluster summary saved to {summary_path}")
        
        # Create cluster details directory
        clusters_dir = os.path.join(self.output_dir, "document_clusters")
        os.makedirs(clusters_dir, exist_ok=True)
        
        # Save detailed cluster information
        for cluster in self.clusters:
            cluster_id = cluster['cluster_id']
            
            # Create detailed page listing
            page_details = []
            for page in cluster['pages']:
                page_details.append({
                    'pdf_file': page['pdf_file'],
                    'page_number': page['page_number'],
                    'text_length': page.get('text_length', 0),
                })
                
            # Save detailed cluster information
            details_df = pd.DataFrame(page_details)
            details_path = os.path.join(clusters_dir, f"cluster_{cluster_id}.csv")
            details_df.to_csv(details_path, index=False)
            
        return summary_path

def main():
    parser = argparse.ArgumentParser(description='Simple Document Clustering')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    parser.add_argument('--sample', '-s', type=int, help='Number of PDFs to process (default: all)')
    
    args = parser.parse_args()
    
    # Verify input directory exists
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return
    
    # Create clustering object
    clustering = SimpleDocumentClustering(args.input, args.output, args.sample)
    
    # Track execution time
    start_time = time.time()
    
    # Execute processing pipeline
    try:
        clustering.process_pdfs()
        clustering.cluster_documents()
        summary_path = clustering.export_results()
        
        # Display total time
        total_time = time.time() - start_time
        time_formatted = str(timedelta(seconds=int(total_time)))
        
        logger.info(f"Processing completed in {time_formatted}")
        logger.info(f"Results available at {args.output}")
        logger.info(f"Summary report: {summary_path}")
        
        print(f"\nProcessing completed in {time_formatted}")
        print(f"Results available at: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"\nERROR: Processing failed: {e}")

if __name__ == "__main__":
    main()