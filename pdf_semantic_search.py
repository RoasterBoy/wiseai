#!/usr/bin/env python3
import argparse
import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

def extract_text_from_pdf(pdf_path):
    """
    Extract full text and per-page text from a PDF file.
    
    Returns:
      full_text: concatenated text of all pages.
      pages: a list of strings where each element corresponds to a page's text.
    """
    full_text = ""
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
            full_text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return full_text, pages

def index_pdfs(pdf_dir, model=None):
    """
    Scan the given directory for PDF files, extract their text (full text and per-page),
    and optionally compute embeddings.
    
    Returns:
      docs: list of full document texts.
      filenames: list of PDF file paths.
      pages_list: list of per-document lists containing page texts.
      embeddings: tensor of embeddings if a model is provided; otherwise None.
    """
    docs = []
    filenames = []
    pages_list = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            full_text, pages = extract_text_from_pdf(pdf_path)
            if full_text.strip():
                docs.append(full_text)
                filenames.append(pdf_path)
                pages_list.append(pages)
            else:
                print(f"No text found in {pdf_path}")
    if not docs:
        print("No valid PDF documents found for indexing.")
        exit(1)
    if model is not None:
        embeddings = model.encode(docs, convert_to_tensor=True)
    else:
        embeddings = None
    return docs, filenames, pages_list, embeddings

def save_index(index_file, docs, filenames, pages_list, embeddings):
    """Save the index data to a file."""
    with open(index_file, "wb") as f:
        pickle.dump({
            "docs": docs,
            "filenames": filenames,
            "pages_list": pages_list,
            "embeddings": embeddings
        }, f)

def load_index(index_file):
    """Load the index data from a file."""
    with open(index_file, "rb") as f:
        index_data = pickle.load(f)
    return (index_data["docs"],
            index_data["filenames"],
            index_data["pages_list"],
            index_data.get("embeddings", None))

def semantic_search(query, docs, filenames, embeddings, model, top_k=3):
    """
    Compute the query embedding and perform cosine similarity search against PDF embeddings.
    
    Results are printed in descending order of similarity score.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_scores, top_indices = cos_scores.topk(k=top_k)
    
    print("\nTop semantic search results (ranked by score):")
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
        print(f"Rank {rank}: File: {filenames[idx]} (Score: {score:.4f})")
        snippet = docs[idx][:200].replace('\n', ' ')
        print(f"Snippet: {snippet}...")
        print("-" * 80)

def text_search(query, docs, filenames, pages_list, top_k=3):
    """
    Perform a simple case-insensitive text search for a given query over the documents.
    
    For each PDF, the function iterates over its pages to identify the page number(s)
    where the query occurs. Each result is ranked by the total number of occurrences.
    """
    query_lower = query.lower()
    results = []
    # For each document, count total occurrences and capture per-page matches.
    for doc, fname, pages in zip(docs, filenames, pages_list):
        total_occurrences = 0
        page_matches = []
        for i, page_text in enumerate(pages, start=1):
            page_text_lower = page_text.lower()
            count = page_text_lower.count(query_lower)
            if count > 0:
                total_occurrences += count
                first_index = page_text_lower.find(query_lower)
                snippet_start = max(0, first_index - 50)
                snippet_end = first_index + 50 + len(query)
                snippet = page_text[snippet_start:snippet_end].replace('\n', ' ')
                page_matches.append((i, count, snippet))
        if total_occurrences > 0:
            results.append((fname, total_occurrences, page_matches))
    
    # Sort results by total_occurrences (highest first).
    results.sort(key=lambda x: x[1], reverse=True)
    
    if results:
        print("\nTop text search results (ranked by total occurrence count):")
        rank = 1
        for fname, total_count, page_matches in results[:top_k]:
            print(f"Rank {rank}: File: {fname} (Total Occurrences: {total_count})")
            for page_num, count, snippet in page_matches:
                print(f"  Page {page_num} - Occurrences: {count}")
                print(f"    Snippet: {snippet}...")
            print("-" * 80)
            rank += 1
    else:
        print("No matches found for the query.")

def main():
    parser = argparse.ArgumentParser(
        description="Semantic or Text Search in OCR'd PDF Files with Index Preservation, "
                    "including Page Numbers for Text Matches"
    )
    parser.add_argument("pdf_dir", type=str, help="Directory containing PDF files")
    parser.add_argument("--query", type=str, help="Search query string", default=None)
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to show")
    parser.add_argument("--index_file", type=str, default="pdf_index.pkl", help="Path to store/load the index file")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing of PDF files")
    parser.add_argument(
        "--search_type",
        type=str,
        choices=["semantic", "text"],
        default="semantic",
        help="Type of search: 'semantic' for semantic search or 'text' for a simple text string search"
    )
    args = parser.parse_args()

    # Load the sentence transformer model only if using semantic search.
    if args.search_type == "semantic":
        print("Loading model for semantic search...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")
    else:
        model = None

    # Load or build the index.
    if not args.reindex and os.path.exists(args.index_file):
        print(f"Loading index from {args.index_file}...")
        docs, filenames, pages_list, embeddings = load_index(args.index_file)
        print(f"Loaded index with {len(docs)} documents.")
    else:
        print("Indexing PDFs...")
        docs, filenames, pages_list, embeddings = index_pdfs(args.pdf_dir, model)
        print(f"Indexed {len(docs)} documents.")
        print(f"Saving index to {args.index_file}...")
        save_index(args.index_file, docs, filenames, pages_list, embeddings)
        print("Index saved.")

    # Execute the appropriate search.
    if args.query:
        if args.search_type == "semantic":
            semantic_search(args.query, docs, filenames, embeddings, model, top_k=args.top_k)
        else:
            text_search(args.query, docs, filenames, pages_list, top_k=args.top_k)
    else:
        print("Enter your search query (or type 'exit' to quit):")
        while True:
            query = input("Query: ")
            if query.lower() in ['exit', 'quit']:
                break
            if args.search_type == "semantic":
                semantic_search(query, docs, filenames, embeddings, model, top_k=args.top_k)
            else:
                text_search(query, docs, filenames, pages_list, top_k=args.top_k)

if __name__ == '__main__':
    main()