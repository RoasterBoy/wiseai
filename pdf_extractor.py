import argparse
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

def extract_text_from_pdf(pdf_path, output_path):
    """
    Extracts text from a PDF file and writes it to a specified output file.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_path (str): The path to the output text file.
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    try:
        with open(pdf_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, 
                                          caching=True,
                                          check_extractable=True):
                page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text)

    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        converter.close()
        fake_file_handle.close()

def main():
    """
    Main function to handle command-line arguments and execute the PDF text extraction.
    """
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument("input_pdf", help="The input PDF file path.")
    parser.add_argument("output_txt", help="The output text file path.")

    args = parser.parse_args()

    extract_text_from_pdf(args.input_pdf, args.output_txt)
    print(f"Text extracted from '{args.input_pdf}' and saved to '{args.output_txt}'.")

if __name__ == "__main__":
    main()