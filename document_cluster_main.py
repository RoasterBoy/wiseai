"""
Document Clustering System - Main Module

This is the main script that coordinates the document clustering process.
It uses separate modules for PDF processing, text extraction, and clustering.
"""

import os
import argparse
import time
from datetime import timedelta
import logging
import traceback

# Import our modules
from doc_cluster.pdf_processor import PDFProcessor
from doc_cluster.text_extractor import TextExtractor
from doc_cluster.document_clusterer import DocumentClusterer
from doc_cluster.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentClusteringSystem:
    def __init__(self, input_dir, output_dir, sample_size=None, 
                 save_enhanced_images=False, save_text_output=True, 
                 max_workers=None):
        """
        Initialize the document clustering system.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save results
            sample_size: Number of PDFs to process (None for all)
            save_enhanced_images: Whether to save preprocessed images
            save_text_output: Whether to save extracted text files
            max_workers: Maximum number of worker processes
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.save_enhanced_images = save_enhanced_images
        self.save_text_output = save_text_output
        self.max_workers = max_workers if max_workers else min(os.cpu_count(), 4)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        if self.save_enhanced_images:
            os.makedirs(os.path.join(output_dir, "enhanced_images"), exist_ok=True)
        
        if self.save_text_output:
            os.makedirs(os.path.join(output_dir, "text_output"), exist_ok=True)
            
        os.makedirs(os.path.join(output_dir, "document_clusters"), exist_ok=True)
        
        # Initialize components
        self.pdf_processor = PDFProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            max_workers=self.max_workers
        )
        
        self.text_extractor = TextExtractor(
            output_dir=output_dir,
            save_enhanced_images=save_enhanced_images,
            save_text_output=save_text_output
        )
        
        self.clusterer = DocumentClusterer()
        self.report_generator = ReportGenerator(output_dir=output_dir)
        
        logger.info(f"Initialized DocumentClusteringSystem with {self.max_workers} workers")
    
    def run(self):
        """
        Run the complete document clustering pipeline.
        
        Returns:
            Path to the summary report file
        """
        start_time = time.time()
        
        try:
            # Step 1: Process PDFs and extract text
            pdf_files, page_texts, page_metadata = self.pdf_processor.process_documents(
                text_extractor=self.text_extractor,
                sample_size=self.sample_size
            )
            
            if not page_texts:
                logger.warning("No text content extracted from documents")
                return self.report_generator.generate_empty_report(pdf_files)
            
            # Step 2: Cluster documents
            document_clusters = self.clusterer.cluster_documents(page_texts, page_metadata)
            
            if not document_clusters:
                logger.warning("No document clusters identified")
                return self.report_generator.generate_empty_report(pdf_files)
            
            # Step 3: Generate reports
            summary_path = self.report_generator.generate_reports(document_clusters, pdf_files)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            time_formatted = str(timedelta(seconds=int(total_time)))
            
            logger.info(f"Complete processing pipeline finished in {time_formatted}")
            logger.info(f"Results available at {self.output_dir}")
            logger.info(f"Summary report: {summary_path}")
            
            return summary_path
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            logger.debug(traceback.format_exc())
            
            # Create an error report
            error_path = os.path.join(self.output_dir, "error_report.txt")
            with open(error_path, "w") as f:
                f.write(f"Error during document clustering: {e}\n\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
            
            logger.info(f"Error report saved to {error_path}")
            return error_path

def main():
    """Command line interface for the document clustering system."""
    parser = argparse.ArgumentParser(description='Document Clustering System')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing PDF files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    parser.add_argument('--sample', '-s', type=int, help='Number of PDFs to process (default: all)')
    parser.add_argument('--workers', '-w', type=int, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--save-images', action='store_true', help='Save enhanced images (default: False)')
    parser.add_argument('--save-text', action='store_true', help='Save extracted text (default: True)')
    
    args = parser.parse_args()
    
    # Create and run the clustering system
    system = DocumentClusteringSystem(
        input_dir=args.input,
        output_dir=args.output,
        sample_size=args.sample,
        save_enhanced_images=args.save_images,
        save_text_output=args.save_text,
        max_workers=args.workers
    )
    
    # Run the pipeline
    summary_path = system.run()
    
    print(f"\nProcessing complete")
    print(f"Results available at: {args.output}")
    print(f"Summary report: {summary_path}")

if __name__ == "__main__":
    main()