from PIL import Image
import pytesseract

# Path to the image
image_path = "IMG_5362.PNG"

# Perform OCR
extracted_text = pytesseract.image_to_string(Image.open(image_path))

print("Extracted Text:")
print(extracted_text)
