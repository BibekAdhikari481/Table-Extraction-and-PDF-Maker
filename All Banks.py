from pdf2image import convert_from_path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image
from fpdf import FPDF
import os



def pdf_to_images_in_memory(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=r"C:\Users\CT_USER\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin")
    images = []
    
    for page in pages:
        img_byte_arr = io.BytesIO()
        page.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        images.append(Image.open(img_byte_arr))
    
    return images



def classify_pages_update(model, images):
    image_arrays = []
    for img in images:
        img_array = np.array(img)
        img_array = tf.image.resize(img_array, [224, 224])
        img_array = img_array / 255.0 
        img_array = np.expand_dims(img_array, axis=0) 
        image_arrays.append(img_array)
    
    image_arrays = np.concatenate(image_arrays, axis=0)
    predictions = model.predict(image_arrays)
    
    return predictions > 0.5



def load_model_for_bank(bank_name):
    model_path = f"D:/PDF Data Extraction/Bankwise Models/{bank_name}_model.h5"
    model = load_model(model_path)
    print(f"Model for {bank_name} loaded.")
    return model



def convert_images_to_pdf_update(images, output_pdf_path):
    pdf = FPDF()

    for idx, image in enumerate(images):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            image = image.copy()
        else:
            raise TypeError("Expected a file path or a PIL Image object")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        temp_file = f"temp_{idx}.jpg"
        image.save(temp_file)

        pdf.add_page()
        pdf.image(temp_file, 0, 0, 210, 297)
        os.remove(temp_file)

    pdf.output(output_pdf_path, "F")



def generate_pdf_with_tables_for_bank(bank_name, pdf_path):
    model = load_model_for_bank(bank_name)
    image_paths = pdf_to_images_in_memory(pdf_path)
    predictions = classify_pages_update(model, image_paths)
    
    filtered_image_paths = []
    for image, contains_table in zip(image_paths, predictions):
        if contains_table:
            filtered_image_paths.append(image)
            
    output_pdf_path = f"D:/PDF Data Extraction/OUTPUT PDF/{bank_name}_tables_only.pdf"
    convert_images_to_pdf_update(filtered_image_paths, output_pdf_path)
    print(f"PDF with tables saved as {output_pdf_path}")




banks = ['Hsbc', 'Barclays', 'Santander', 'Royal Bank of Scotland', 'Metro','Revolut','Paypal','Natwest','Nationwide']
bank=input("Enter bank name:")

if bank in banks:
    generate_pdf_with_tables_for_bank(bank, r"D:/PDF Data Extraction/Metro Merged.pdf")
    print("Pdf saved!")
else:
    print("Please enter a valid bank name with correct spelling!")