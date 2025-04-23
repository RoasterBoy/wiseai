#!/usr/bin/env python3
"""
Secondary analyzer for documents categorized as "Other"
Identifies subcategories and generates reports and visualizations
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Secondary analysis for "Other" documents')
    parser.add_argument('--input', '-i', required=True, help='Input directory with processed documents')
    parser.add_argument('--summary', '-s', required=True, help='Path to document summary CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--list-files', '-f', action='store_true', help='Generate file listings by subcategory')
    parser.add_argument('--num-clusters', '-n', type=int, default=5, help='Number of subcategories to identify')
    parser.add_argument('--top-keywords', '-k', type=int, default=10, help='Number of top keywords to extract')
    parser.add_argument('--category-column', type=str, default='document_type', 
                       help='Column name for category (default: document_type)')
    parser.add_argument('--other-value', type=str, default='Other', 
                       help='Value that represents "Other" category (default: Other)')
    parser.add_argument('--text-subdir', type=str, default='text_output', 
                       help='Subdirectory containing text files (default: text_output)')
    return parser.parse_args()

def extract_document_id(file_path):
    """Extract document ID from file path"""
    # Remove the text_output part and any trailing separators
    path_parts = file_path.split('/')
    
    # Look for the folder containing text_output
    for i in range(len(path_parts) - 1):
        if path_parts[i] == 'text_output':
            # Return the parent folder name
            return path_parts[i-1]
    
    # If text_output not found, use the first folder after 'clusters'
    for i in range(len(path_parts) - 1):
        if path_parts[i] == 'clusters' and i < len(path_parts) - 1:
            return path_parts[i+1]
    
    # If all else fails, return the last part of the path
    return os.path.basename(file_path)

def load_documents_from_summary(summary_file, input_dir, text_subdir='text_output', 
                                category_column='document_type', other_value='Other', verbose=False):
    """Load documents using paths from the summary file"""
    if verbose:
        print(f"Loading documents from summary file: {summary_file}")
    
    try:
        df_summary = pd.read_csv(summary_file)
        if verbose:
            print(f"Loaded summary file with {len(df_summary)} entries")
            print(f"Available columns: {', '.join(df_summary.columns)}")
    except Exception as e:
        print(f"Error loading summary file: {e}")
        sys.exit(1)
    
    # Check if category column exists
    if category_column not in df_summary.columns:
        print(f"Error: Column '{category_column}' not found in summary file.")
        print(f"Available columns are: {', '.join(df_summary.columns)}")
        print("Please specify the correct column name using --category-column.")
        sys.exit(1)
    
    # Filter to only include "Other" category documents
    df_other = df_summary[df_summary[category_column] == other_value]
    if verbose:
        print(f"Found {len(df_other)} documents in '{other_value}' category")
    
    # If no documents found, try to process all documents
    if len(df_other) == 0:
        print(f"Warning: No documents found with '{category_column}' = '{other_value}'")
        print("Processing all documents instead.")
        df_other = df_summary
    
    # Check if text_file column exists
    text_file_column = None
    for col in ['text_file', 'file_path', 'filepath', 'path', 'file']:
        if col in df_other.columns:
            text_file_column = col
            break
    
    if text_file_column is None:
        print("Error: Could not find column with file paths in summary file.")
        print(f"Available columns: {', '.join(df_other.columns)}")
        sys.exit(1)
    
    if verbose:
        print(f"Using column '{text_file_column}' for file paths")
    
    # Load document content
    documents = []
    filenames = []
    doc_ids = []
    
    for idx, row in df_other.iterrows():
        # Get file path
        file_path = str(row[text_file_column])
        
        # Check if file exists directly
        if os.path.exists(file_path) and os.path.isfile(file_path):
            full_path = file_path
        else:
            # Check if file exists relative to input_dir
            relative_path = os.path.join(input_dir, file_path)
            if os.path.exists(relative_path) and os.path.isfile(relative_path):
                full_path = relative_path
            else:
                # Try to find the file by extracting document ID and looking in text_subdir
                doc_id = extract_document_id(file_path)
                doc_dir = os.path.join(input_dir, doc_id)
                
                # Check if base text file exists
                base_name = os.path.basename(file_path)
                text_file_path = os.path.join(doc_dir, text_subdir, base_name)
                
                if os.path.exists(text_file_path) and os.path.isfile(text_file_path):
                    full_path = text_file_path
                else:
                    # Try finding any text file in this directory
                    if os.path.exists(doc_dir) and os.path.isdir(doc_dir):
                        text_dir = os.path.join(doc_dir, text_subdir)
                        if os.path.exists(text_dir) and os.path.isdir(text_dir):
                            text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
                            if text_files:
                                full_path = os.path.join(text_dir, text_files[0])
                            else:
                                if verbose:
                                    print(f"Warning: No text files found in {text_dir}")
                                continue
                        else:
                            if verbose:
                                print(f"Warning: Text directory not found: {text_dir}")
                            continue
                    else:
                        if verbose:
                            print(f"Warning: Document directory not found: {doc_dir}")
                        continue
        
        # Try to read the file
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():  # Only add non-empty documents
                    documents.append(content)
                    filenames.append(full_path)
                    doc_ids.append(extract_document_id(full_path))
                    if verbose and len(documents) % 20 == 0:
                        print(f"Loaded {len(documents)} documents...")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read {full_path}: {e}")
    
    if verbose:
        print(f"Successfully loaded {len(documents)} documents")
    
    return documents, filenames, doc_ids, df_other

def find_all_text_files(input_dir, text_subdir='text_output', verbose=False):
    """Find all text files in the directory structure"""
    all_files = []
    
    if verbose:
        print(f"Searching for all text files in {input_dir}")
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(input_dir):
        # Check if we're in a text_output directory
        if os.path.basename(root) == text_subdir:
            for file in files:
                if file.endswith('.txt'):
                    filepath = os.path.join(root, file)
                    doc_id = extract_document_id(filepath)
                    all_files.append((doc_id, filepath))
    
    if verbose:
        print(f"Found {len(all_files)} text files")
    
    return all_files

def load_documents(input_dir, summary_file, verbose=False, category_column='document_type', 
                  other_value='Other', text_subdir='text_output'):
    """Load document data from the input directory and summary file"""
    if verbose:
        print(f"Loading documents from {input_dir}")
        print(f"Using summary file: {summary_file}")
    
    # First try to load documents from summary file
    documents, filenames, doc_ids, df_other = load_documents_from_summary(
        summary_file, input_dir, text_subdir, category_column, other_value, verbose
    )
    
    # If no documents loaded, try to find all text files
    if not documents:
        if verbose:
            print("No documents loaded from summary file. Finding all text files...")
        
        all_files = find_all_text_files(input_dir, text_subdir, verbose)
        
        for doc_id, filepath in all_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty documents
                        documents.append(content)
                        filenames.append(filepath)
                        doc_ids.append(doc_id)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not read {filepath}: {e}")
        
        if verbose:
            print(f"Loaded {len(documents)} documents from filesystem")
    
    return documents, filenames, doc_ids, df_other

def identify_subcategories(documents, filenames, num_clusters=5, verbose=False):
    """Identify subcategories using clustering"""
    if not documents:
        print("Error: No documents to analyze")
        sys.exit(1)
        
    if verbose:
        print(f"Identifying {num_clusters} subcategories...")
    
    # Adjust number of clusters based on document count
    if len(documents) < num_clusters:
        old_num_clusters = num_clusters
        num_clusters = max(2, min(len(documents), 3))  # At least 2, at most 3 clusters for small document sets
        if verbose:
            print(f"Too few documents ({len(documents)}) for {old_num_clusters} clusters. Using {num_clusters} instead.")
    
    # Create TF-IDF vectors
    if verbose:
        print("Generating TF-IDF vectors")
    
    # Adjust parameters based on document count
    min_df = 1 if len(documents) < 10 else 2
    max_df = 1.0 if len(documents) < 10 else 0.9
    
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        stop_words='english',
        use_idf=True
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        if verbose:
            print(f"Created TF-IDF matrix with shape {tfidf_matrix.shape}")
    except Exception as e:
        print(f"Error in TF-IDF vectorization: {e}")
        sys.exit(1)
    
    # Handle the case with very few documents
    if len(documents) <= 3:
        print("Too few documents for meaningful clustering. Classifying all as 'miscellaneous'.")
        results = [(filename, 'miscellaneous', 0) for filename in filenames]
        subcategory_names = {0: 'miscellaneous'}
        cluster_keywords = {0: ['document', 'file', 'miscellaneous']}
        clusters = [0] * len(documents)
        return results, subcategory_names, cluster_keywords, clusters
    
    # Perform clustering
    if verbose:
        print(f"Performing KMeans clustering with {num_clusters} clusters")
    
    try:
        km = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = km.fit_predict(tfidf_matrix)
        if verbose:
            print("Clustering complete")
    except Exception as e:
        print(f"Error in clustering: {e}")
        sys.exit(1)
    
    # Extract keywords for each cluster
    if verbose:
        print("Extracting keywords for each cluster")
    
    cluster_keywords = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(num_clusters):
        # Get documents in this cluster
        cluster_docs = [idx for idx, label in enumerate(clusters) if label == i]
        
        if not cluster_docs:
            # Empty cluster - assign default keywords
            cluster_keywords[i] = ['miscellaneous', 'uncategorized', 'other']
            continue
            
        # Get top terms for this cluster
        cluster_center = km.cluster_centers_[i]
        ordered_centroids = cluster_center.argsort()[::-1]
        
        # Get top keywords, filtering out very short terms
        keywords = []
        for idx in ordered_centroids:
            term = feature_names[idx]
            if len(term) > 2:  # Skip very short terms
                keywords.append(term)
            if len(keywords) >= 10:
                break
                
        cluster_keywords[i] = keywords[:10]  # Ensure we have at most 10 keywords
        
        if verbose:
            print(f"Cluster {i}: {', '.join(cluster_keywords[i][:5])}")
    
    # Assign subcategory names based on keywords
    subcategory_names = {}
    for cluster_id, keywords in cluster_keywords.items():
        if keywords:
            # Use the first meaningful keyword
            subcategory_names[cluster_id] = keywords[0]
        else:
            subcategory_names[cluster_id] = f'category_{cluster_id}'
    
    # Ensure there's a "miscellaneous" category
    has_misc = False
    for name in subcategory_names.values():
        if name in ['misc', 'miscellaneous', 'other', 'unknown']:
            has_misc = True
            break
    
    if not has_misc:
        # Assign the smallest cluster as miscellaneous
        counts = Counter(clusters)
        smallest_cluster = min(counts, key=counts.get)
        subcategory_names[smallest_cluster] = 'miscellaneous'
    
    # Create a mapping of documents to subcategories
    results = []
    for i, (doc, filename) in enumerate(zip(documents, filenames)):
        cluster_id = clusters[i]
        subcategory = subcategory_names[cluster_id]
        results.append((filename, subcategory, cluster_id))
    
    return results, subcategory_names, cluster_keywords, clusters

def generate_report(results, subcategory_names, cluster_keywords, doc_ids, output_dir, verbose=False):
    """Generate report files and visualizations"""
    if verbose:
        print(f"Generating reports in {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results, columns=['filepath', 'subcategory', 'cluster_id'])
    df_results['document_id'] = doc_ids
    
    # Generate subcategory counts
    subcategory_counts = df_results['subcategory'].value_counts()
    counts_file = os.path.join(output_dir, 'subcategory_counts.csv')
    subcategory_counts.to_csv(counts_file)
    
    if verbose:
        print(f"Saved subcategory counts to {counts_file}")
    
    # Generate keyword file
    keyword_data = []
    for cluster_id, keywords in cluster_keywords.items():
        subcategory = subcategory_names[cluster_id]
        keyword_str = ', '.join(keywords)
        keyword_data.append([subcategory, keyword_str])
    
    df_keywords = pd.DataFrame(keyword_data, columns=['subcategory', 'keywords'])
    keyword_file = os.path.join(output_dir, 'subcategory_keywords.csv')
    df_keywords.to_csv(keyword_file, index=False)
    
    if verbose:
        print(f"Saved subcategory keywords to {keyword_file}")
    
    # Generate file listings by subcategory
    files_by_category = df_results[['document_id', 'filepath', 'subcategory']]
    files_output = os.path.join(output_dir, 'files_by_subcategory.csv')
    files_by_category.to_csv(files_output, index=False)
    
    if verbose:
        print(f"Saved files by subcategory to {files_output}")
    
    # Generate a separate file for miscellaneous category
    misc_files = df_results[df_results['subcategory'] == 'miscellaneous']
    misc_output = os.path.join(output_dir, 'miscellaneous_files.txt')
    with open(misc_output, 'w') as f:
        f.write(f"# Miscellaneous Files ({len(misc_files)} total)\n")
        f.write("# These files could not be categorized into specific subcategories\n\n")
        for _, row in misc_files.iterrows():
            f.write(f"{row['document_id']}: {row['filepath']}\n")
    
    if verbose:
        print(f"Saved miscellaneous files to {misc_output}")
        
    # Generate visualizations
    try:
        # Bar chart
        plt.figure(figsize=(10, 6))
        subcategory_counts.plot(kind='bar')
        plt.title('Distribution of Subcategories')
        plt.xlabel('Subcategory')
        plt.ylabel('Count')
        plt.tight_layout()
        bar_chart_file = os.path.join(output_dir, 'subcategories_bar.png')
        plt.savefig(bar_chart_file)
        
        # Pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(subcategory_counts, labels=subcategory_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Subcategories')
        plt.tight_layout()
        pie_chart_file = os.path.join(output_dir, 'subcategories_pie.png')
        plt.savefig(pie_chart_file)
        
        if verbose:
            print(f"Generated visualizations: {bar_chart_file}, {pie_chart_file}")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
    
    return counts_file, keyword_file, files_output

def generate_file_listing(results, doc_ids, input_dir, output_dir, text_subdir='text_output', verbose=False):
    """Generate detailed file listings and previews by subcategory"""
    if verbose:
        print("Generating detailed file listing...")
    
    # Create a directory of files by subcategory
    subcategory_dirs = {}
    for _, subcategory, _ in results:
        if subcategory not in subcategory_dirs:
            subcat_dir = os.path.join(output_dir, subcategory)
            os.makedirs(subcat_dir, exist_ok=True)
            subcategory_dirs[subcategory] = subcat_dir
    
    # Create symbolic links to original files
    for i, (filepath, subcategory, _) in enumerate(results):
        doc_id = doc_ids[i]
        
        # Identify the source file
        if os.path.exists(filepath) and os.path.isfile(filepath):
            src_path = filepath
        else:
            src_path = None
            # Try to find the file in the text_output directory
            possible_dir = os.path.join(input_dir, doc_id, text_subdir)
            if os.path.exists(possible_dir) and os.path.isdir(possible_dir):
                txt_files = [f for f in os.listdir(possible_dir) if f.endswith('.txt')]
                if txt_files:
                    src_path = os.path.join(possible_dir, txt_files[0])
        
        if src_path and os.path.exists(src_path):
            # Create a descriptive name for the symlink
            link_name = f"{doc_id}_{os.path.basename(src_path)}"
            dst_path = os.path.join(subcategory_dirs[subcategory], link_name)
            
            try:
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                os.symlink(os.path.abspath(src_path), dst_path)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not create symlink for {filepath}: {e}")
    
    if verbose:
        print(f"Created file directories by subcategory in {output_dir}")
    
    # Generate a content preview for miscellaneous files
    misc_files = [(filepath, doc_id) for i, (filepath, subcategory, _) in enumerate(results) 
                 if subcategory == 'miscellaneous' and i < len(doc_ids) and doc_ids[i]]
    
    if misc_files:
        misc_preview_file = os.path.join(output_dir, 'miscellaneous_preview.txt')
        with open(misc_preview_file, 'w') as f:
            f.write(f"# Preview of {len(misc_files)} Miscellaneous Files\n")
            f.write("="*80 + "\n\n")
            
            for filepath, doc_id in misc_files:
                # Try to open the file
                readable_path = filepath
                if not os.path.exists(filepath) or not os.path.isfile(filepath):
                    # Try to find the file in the text_output directory
                    possible_dir = os.path.join(input_dir, doc_id, text_subdir)
                    if os.path.exists(possible_dir) and os.path.isdir(possible_dir):
                        txt_files = [f for f in os.listdir(possible_dir) if f.endswith('.txt')]
                        if txt_files:
                            readable_path = os.path.join(possible_dir, txt_files[0])
                
                f.write(f"## File: {doc_id}\n")
                f.write("-"*80 + "\n")
                
                if os.path.exists(readable_path) and os.path.isfile(readable_path):
                    try:
                        with open(readable_path, 'r', encoding='utf-8') as src_file:
                            # Write first 20 lines or 500 characters as preview
                            content = src_file.read(500)
                            lines = content.split('\n')[:20]
                            preview = '\n'.join(lines)
                            f.write(preview)
                            f.write("\n\n" + "="*80 + "\n\n")
                    except Exception as e:
                        f.write(f"Could not read file: {e}\n\n")
                else:
                    f.write(f"File not found: {readable_path}\n\n")
                    f.write("="*80 + "\n\n")
        
        if verbose:
            print(f"Generated miscellaneous file preview: {misc_preview_file}")

def main():
    """Main function"""
    args = parse_arguments()
    
    if args.verbose:
        print("Starting secondary analysis...")
    
    # Load documents
    documents, filenames, doc_ids, df_other = load_documents(
        args.input, 
        args.summary, 
        args.verbose, 
        args.category_column, 
        args.other_value,
        args.text_subdir
    )
    
    if not documents:
        print("No documents found to analyze")
        return 1
    
    # Identify subcategories
    results, subcategory_names, cluster_keywords, clusters = identify_subcategories(
        documents, filenames, args.num_clusters, args.verbose
    )
    
    # Generate report
    counts_file, keyword_file, files_output = generate_report(
        results, subcategory_names, cluster_keywords, doc_ids, args.output, args.verbose
    )
    
    # Generate detailed file listing if requested
    if args.list_files:
        generate_file_listing(
            results, doc_ids, args.input, args.output, args.text_subdir, args.verbose
        )
    
    if args.verbose:
        print("Secondary analysis complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())