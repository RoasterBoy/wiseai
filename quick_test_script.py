import os
import sys
import subprocess
import platform
from pathlib import Path

print("===== PDF Processing Diagnostic Tool =====")
print(f"Python version: {sys.version}")
print(f"Operating system: {platform.system()} {platform.release()}")
print()

# Check Tesseract installation
print("Checking Tesseract OCR installation...")
try:
    version_output = subprocess.check_output(['tesseract', '--version'], stderr=subprocess.STDOUT, text=True)
    print(f"✓ Tesseract is installed: {version_output.splitlines()[0]}")
    
    # Try to get available languages
    langs_output = subprocess.check_output(['tesseract', '--list-langs'], stderr=subprocess.STDOUT, text=True)
    print(f"✓ Available languages: {langs_output}")
except FileNotFoundError:
    print("✗ Tesseract executable not found in PATH")
    print("  Please install Tesseract OCR or add it to your PATH")
    if platform.system() == "Windows":
        print("  Windows installation: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Then set pytesseract.pytesseract.tesseract_cmd in your script")
    else:
        print("  Linux installation: sudo apt install tesseract-ocr tesseract-ocr-eng")
except Exception as e:
    print(f"✗ Error checking Tesseract: {e}")

# Check for pdf2image dependencies
print("\nChecking pdf2image dependencies...")
if platform.system() == "Windows":
    print("On Windows, you need poppler installed:")
    print("  Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
    print("  Add bin directory to PATH or specify poppler_path in your code")
    
    # Try to import pdf2image
    try:
        from pdf2image import convert_from_path
        print("✓ pdf2image module is installed")
    except ImportError:
        print("✗ pdf2image module is not installed")
        print("  Install with: pip install pdf2image")
else:
    # For Linux/Mac, check for poppler-utils
    try:
        poppler_output = subprocess.check_output(['pdftoppm', '-v'], stderr=subprocess.STDOUT, text=True)
        print(f"✓ poppler-utils is installed: {poppler_output.splitlines()[0]}")
    except FileNotFoundError:
        print("✗ poppler-utils not found")
        print("  Install with: sudo apt install poppler-utils")
    except Exception as e:
        print(f"✗ Error checking poppler-utils: {e}")
    
    # Try to import pdf2image
    try:
        from pdf2image import convert_from_path
        print("✓ pdf2image module is installed")
    except ImportError:
        print("✗ pdf2image module is not installed")
        print("  Install with: pip install pdf2image")

# Check other required Python packages
print("\nChecking other required Python packages...")
required_packages = [
    "pytesseract", "numpy", "opencv-python", "scikit-learn", "pandas"
]

for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is not installed")
        print(f"  Install with: pip install {package}")

# Check input/output directories
print("\nChecking directory access...")
try:
    # Get input directory from user
    input_dir = input("Enter the input directory path: ")
    if not os.path.isdir(input_dir):
        print(f"✗ Input directory does not exist: {input_dir}")
    else:
        print(f"✓ Input directory exists: {input_dir}")
        # Check for PDF files
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        print(f"  Found {len(pdf_files)} PDF files")
        if pdf_files:
            print(f"  First PDF file: {pdf_files[0]}")
            # Check if we can read the file
            try:
                with open(os.path.join(input_dir, pdf_files[0]), 'rb') as f:
                    f.read(1024)  # Read first 1KB
                print(f"✓ Can read PDF file: {pdf_files[0]}")
            except Exception as e:
                print(f"✗ Cannot read PDF file: {e}")
    
    # Get output directory from user
    output_dir = input("Enter the output directory path: ")
    if not os.path.isdir(output_dir):
        print(f"✗ Output directory does not exist: {output_dir}")
        create = input("Create output directory? (y/n): ")
        if create.lower() == 'y':
            try:
                os.makedirs(output_dir)
                print(f"✓ Created output directory: {output_dir}")
            except Exception as e:
                print(f"✗ Error creating output directory: {e}")
    else:
        print(f"✓ Output directory exists: {output_dir}")
        # Check if we can write to it
        try:
            test_file = os.path.join(output_dir, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("Test write access")
            os.remove(test_file)
            print(f"✓ Can write to output directory")
        except Exception as e:
            print(f"✗ Cannot write to output directory: {e}")
            
except Exception as e:
    print(f"Error checking directories: {e}")

print("\n===== Diagnostic Complete =====")
print("If all checks passed, try running the debug script with a single PDF file.")
