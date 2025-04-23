#!/usr/bin/env python3
"""
Document Type Analyzer and Visualizer

This script analyzes processed PDF documents, determines their types,
generates a summary CSV file, and creates visualizations.
"""

import os
import sys
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_analyzer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the document analyzer.
        
        Args:
            input_dir: Directory containing processed documents
            output_dir: Directory to save results
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.document_types = defaultdict(list)
        self.document_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_document_type(self, text_content):
        """
        Analyze text content to determine document type.
        
        Args:
            text_content: Extracted text from document
            
        Returns:
            Document type string
        """
        text_lower = text_content.lower()
        
        # Check for letter patterns
        if re.search(r'dear\s+[a-z,\.]+|sincerely|regards|yours truly', text_lower):
            return 'Letter'
        
        # Check for receipt patterns
        if (re.search(r'receipt|total|subtotal|payment|paid|amount|\$|€|£', text_lower) and 
            re.search(r'\d+[\.,]\d{2}', text_content) and len(text_content) < 1000):
            return 'Receipt'
        
        # Check for report patterns
        if (re.search(r'report|analysis|summary|findings|conclusion|introduction', text_lower) and 
            len(text_content) > 1000):
            return 'Report'
            
        # Check for news clipping patterns
        if re.search(r'newspaper|article|press|published|journalist|editor|news', text_lower):
            return 'News Clipping'
            
        # Check for form patterns
        if re.search(r'form|please fill|checkbox|check one|signature|date of birth', text_lower):
            return 'Form'
            
        # Default
        return 'Other'
    
    def analyze_directory(self):
        """
        Analyze all documents in the input directory.
        """
        logger.info(f"Analyzing documents in: {self.input_dir}")
        
        # Find all subdirectories (each should be a processed document)
        doc_dirs = [d for d in os.listdir(self.input_dir) 
                   if os.path.isdir(os.path.join(self.input_dir, d))]
        
        logger.info(f"Found {len(doc_dirs)} potential document directories")
        
        for doc_dir in doc_dirs:
            dir_path = os.path.join(self.input_dir, doc_dir)
            
            # Look for text output directory
            text_dir = os.path.join(dir_path, "text_output")
            if not os.path.isdir(text_dir):
                logger.warning(f"No text_output directory found in {dir_path}")
                continue
            
            # Get all text files
            text_files = glob.glob(os.path.join(text_dir, "*.txt"))
            if not text_files:
                logger.warning(f"No text files found in {text_dir}")
                continue
            
            # Analyze the first page of text to determine document type
            try:
                with open(text_files[0], 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                # Determine document type
                doc_type = self.analyze_document_type(text_content)
                
                # Store document information
                self.document_types[doc_type].append({
                    'document_name': doc_dir,
                    'text_file': text_files[0],
                    'page_count': len(text_files)
                })
                
                self.document_count += 1
                
                logger.info(f"Classified {doc_dir} as: {doc_type}")
                
            except Exception as e:
                logger.error(f"Error analyzing {text_files[0]}: {e}")
        
        logger.info(f"Analysis complete. Classified {self.document_count} documents")
    
    def generate_summary(self):
        """
        Generate summary CSV and visualization.
        """
        if self.document_count == 0:
            logger.error("No documents were analyzed")
            return False
        
        # Create summary dataframe
        rows = []
        for doc_type, documents in self.document_types.items():
            for doc in documents:
                rows.append({
                    'document_name': doc['document_name'],
                    'document_type': doc_type,
                    'page_count': doc['page_count'],
                    'text_file': doc['text_file']
                })
        
        summary_df = pd.DataFrame(rows)
        
        # Save to CSV
        summary_path = os.path.join(self.output_dir, "document_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved document summary to: {summary_path}")
        
        # Generate type counts
        type_counts = Counter()
        for doc_type, documents in self.document_types.items():
            type_counts[doc_type] = len(documents)
        
        # Save type counts to CSV
        counts_df = pd.DataFrame([
            {'document_type': doc_type, 'count': count}
            for doc_type, count in type_counts.items()
        ])
        counts_path = os.path.join(self.output_dir, "document_type_counts.csv")
        counts_df.to_csv(counts_path, index=False)
        logger.info(f"Saved document type counts to: {counts_path}")
        
        # Create visualizations
        self.create_visualizations(type_counts)
        
        return True
    
    def create_visualizations(self, type_counts):
        """
        Create visualizations of document types.
        
        Args:
            type_counts: Counter of document type counts
        """
        try:
            # Bar chart
            plt.figure(figsize=(10, 6))
            types = list(type_counts.keys())
            counts = [type_counts[t] for t in types]
            
            # Sort by frequency
            sorted_data = sorted(zip(types, counts), key=lambda x: x[1], reverse=True)
            types = [t for t, _ in sorted_data]
            counts = [c for _, c in sorted_data]
            
            # Define colors
            colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe', '#00C49F']
            if len(types) > len(colors):
                colors = colors * (len(types) // len(colors) + 1)
            
            plt.bar(types, counts, color=colors[:len(types)])
            plt.xlabel('Document Type')
            plt.ylabel('Number of Documents')
            plt.title('Document Types in Collection')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save bar chart
            bar_chart_path = os.path.join(self.output_dir, "document_types_bar.png")
            plt.savefig(bar_chart_path)
            logger.info(f"Saved bar chart to: {bar_chart_path}")
            
            # Pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(counts, labels=types, autopct='%1.1f%%', startangle=90, colors=colors[:len(types)])
            plt.axis('equal')
            plt.title('Document Type Distribution')
            
            # Save pie chart
            pie_chart_path = os.path.join(self.output_dir, "document_types_pie.png")
            plt.savefig(pie_chart_path)
            logger.info(f"Saved pie chart to: {pie_chart_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

def main():
    parser = argparse.ArgumentParser(description='Document Type Analyzer')
    parser.add_argument('--input', '-i', required=True, help='Input directory with processed documents')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return 1
    
    # Create analyzer
    analyzer = DocumentAnalyzer(args.input, args.output)
    
    # Run analysis
    analyzer.analyze_directory()
    
    # Generate summary and visualizations
    if analyzer.generate_summary():
        logger.info("Document analysis completed successfully")
        return 0
    else:
        logger.error("Document analysis failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
