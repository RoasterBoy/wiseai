import fitz  # PyMuPDF
import re
import os
import argparse
import csv

def extract_title(text, previous_title=None):
    # Regex for NAME & NAME
    ampersand_pattern = re.compile(r"\b[A-Z]{2,}\s*&\s*[A-Z]{2,}\b")
    # Regex for NAME NAME
    all_caps_name_pattern = re.compile(r"\b[A-Z]{2,}\s+[A-Z]{2,}\b")
    # Regex for Pg 2 or Pg2
    continuation_pattern = re.compile(r"\bPg\s*2\b", re.IGNORECASE)

    if continuation_pattern.search(text):
        return previous_title

    match = ampersand_pattern.search(text)
    if match:
        return match.group()

    match = all_caps_name_pattern.search(text)
    if match:
        return match.group()

    return "Untitled"

def extract_images_with_titles(pdf_path, output_dir, csv_path):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    previous_title = None
    image_count = 0
    empty_pages = 0

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        title = extract_title(text, previous_title)
        if not title:
            title = "Untitled"
        else:
            previous_title = title

        images = page.get_images(full=True)
        if not images:
            print(f"⚠️  Page {page_number}: No images found. Title = '{title}'")
            empty_pages += 1
            continue  # comment this out if you want to log title even without image

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{title.replace(' ', '_')}_{page_number}_{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            metadata.append({
                "page": page_number,
                "image_file": image_filename,
                "title": title
            })
            image_count += 1
            print(f"✅ Page {page_number}: Extracted image {image_filename}")

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["page", "image_file", "title"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\n✅ Done: {image_count} image(s) extracted.")
    print(f"📭 {empty_pages} page(s) had no extractable images.")

    print(f"✅ Extracted {image_count} images and saved metadata to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from a PDF and label them by detected title.")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("output_dir", help="Directory to save extracted images")
    parser.add_argument("csv_file", help="CSV file to store image metadata")

    args = parser.parse_args()

    extract_images_with_titles(args.input_pdf, args.output_dir, args.csv_file)