
import os
import csv
from lxml import etree
import argparse

def extract_figures_with_context(xml_path, output_csv):
    with open(xml_path, "rb") as f:
        tree = etree.parse(f)

    root = tree.getroot()
    results = []

    current_heading = None
    figure_count = 0

    for elem in root.iter():
        if not hasattr(elem.tag, 'startswith'):
            continue  # skip comments or processing instructions

        tag = etree.QName(elem).localname

        if tag in ("H1", "H2"):
            current_heading = elem.text.strip() if elem.text else ""

        elif tag == "Figure":
            # Try to find Alt text or nearby P content (if nested)
            caption = ""
            for child in elem.iter():
                if etree.QName(child).localname == "Alt" and child.text:
                    caption = child.text.strip()
                    break
                elif etree.QName(child).localname == "P" and child.text:
                    caption = child.text.strip()
                    break

            results.append({
                "figure_index": figure_count + 1,
                "heading": current_heading or "Untitled",
                "caption_or_text": caption
            })
            figure_count += 1

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["figure_index", "heading", "caption_or_text"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Extracted {figure_count} figures to '{output_csv}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract figures from Adobe-exported XML and link to nearest headings.")
    parser.add_argument("input_xml", help="Path to Acrobat-exported XML file")
    parser.add_argument("output_csv", help="Path to output CSV file")

    args = parser.parse_args()
    extract_figures_with_context(args.input_xml, args.output_csv)
