import json
import sys

def json_to_html(json_file, html_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(html_file, 'w') as out:
        out.write("<html><body>\n")

        for item in data:
            header = item.get('page_header', '').strip()
            image_src = item.get('extracted_image', '').strip()
            caption = item.get('image_caption', '').strip()
            source = item.get('source_page', '').strip()
            footer = item.get('page_footer', '').strip()

            out.write(f"<h2>{header}</h2>\n")
            out.write("<p>\n")

            # Only include image if present
            if image_src:
                out.write(f'  <img src="{image_src}" width="300" />\n')

            if caption:
                out.write(f'  <br>{caption}\n')
            out.write("</p>\n")

            if source:
                out.write("<p>\n")
                out.write(f'  <a href="{source}">Source File</a>\n')
                out.write("</p>\n")

            if footer:
                out.write(f'<p>Page footer: {footer}</p>\n')

            out.write("<hr>\n")

        out.write("</body></html>\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_json_to_html.py metadata.json output.html")
    else:
        json_to_html(sys.argv[1], sys.argv[2])