import pdfplumber

with pdfplumber.open("Dunton.pdf") as pdf:
    page = pdf.pages[0]
    im = page.to_image()
    im.debug_tablefinder()  # or im.debug_text()
    im.save("page_with_overlays.png")
