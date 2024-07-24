import os
import fitz  # PyMuPDF
from google.cloud import vision
from google.cloud import aiplatform
from PIL import Image
import cv2
import numpy as np

# Initialize the Google Cloud Vision client
project = "drgenai"
location = "us-central1"
aiplatform.init(project=project,location=location)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "drgenai-7daf412ca440.json"
vision_client = vision.ImageAnnotatorClient()

def extract_text_and_barcodes_from_pdf(pdf_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        print(f"Successfully opened PDF with {pdf_document.page_count} pages")
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return []

    unique_barcodes = {}
    extracted_text = ""

    for page_num in range(pdf_document.page_count):
        try:
            # Get the page
            page = pdf_document.load_page(page_num)

            # Get the pixmap (image) of the page
            pix = page.get_pixmap(dpi=300)
            
            # Convert to PIL image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert PIL Image to OpenCV format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert the image to bytes for Google Cloud Vision API
            img_byte_arr = cv2.imencode('.jpg', img_cv)[1].tobytes()
            image = vision.Image(content=img_byte_arr)

            # Perform text detection
            response = vision_client.text_detection(image=image)
            texts = response.text_annotations
            if texts:
                extracted_text += texts[0].description
            
            # Perform barcode detection (limited to some barcode formats)
            response = vision_client.document_text_detection(image=image)
            for entity in response.full_text_annotation.pages:
                for block in entity.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            if word_text.isnumeric() and len(word_text) > 5:  # Simple heuristic for barcodes
                                if word_text not in unique_barcodes:
                                    unique_barcodes[word_text] = {
                                        'type': 'Numeric',
                                        'data': word_text,
                                        'method': 'google-cloud-vision'
                                    }
                                    print(f"Detected barcode: {unique_barcodes[word_text]}")

        except Exception as e:
            print(f"Error processing page {page_num}: {e}")

    return extracted_text, list(unique_barcodes.values())

# Usage
pdf_path = ''
if os.path.exists(pdf_path):
    text, barcodes = extract_text_and_barcodes_from_pdf(pdf_path)
    print("\nExtracted Text:")
    print(text)
    print("\nExtracted Barcodes:")
    for barcode in barcodes:
        print(f"Type: {barcode['type']}, Data: {barcode['data']}, Method: {barcode['method']}")
else:
    print(f"Error: The file {pdf_path} does not exist.")
