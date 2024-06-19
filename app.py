# Importing neccesary libraries
from PIL import Image
import streamlit as st
import easyocr
from ultralytics import YOLO
import shutil
import os

# Setting Webpage Configurations
st.set_page_config(page_icon="📀",page_title="Object Recognition", layout="wide")

# Title
st.title(':red[License Plate Recognition] - YOLO v8 🛡️')

# image upload
image_upload = st.file_uploader('Upload your Image')

if image_upload is not None:
    image = Image.open(image_upload)
    

# submit button
submit = st.button('Upload')

# Loading the model
model = YOLO(r'G:\GUVI_DATA_SCIENCE\Project\Vehicle-License-Plate-Recogition-using-YOLO-v8-and-Easy-OCR\custom_trained_model.pt')

progress_text = "Operation in progress. Please wait."

with st.spinner(progress_text):
    try:
        if submit:
            temporary_image_path = r'G:\GUVI_DATA_SCIENCE\Project\Vehicle-License-Plate-Recogition-using-YOLO-v8-and-Easy-OCR\temp_images'
            image.save(os.path.join(temporary_image_path, 'temp.jpg'))

            model.predict(source = r'temp_images\temp.jpg', save = True, conf = 0.5, classes = 0, save_crop = True)

            bbox_image = r'G:\GUVI_DATA_SCIENCE\Project\Vehicle-License-Plate-Recogition-using-YOLO-v8-and-Easy-OCR\runs\detect\predict\temp.jpg'

            crop_license_image = r'G:\GUVI_DATA_SCIENCE\Project\Vehicle-License-Plate-Recogition-using-YOLO-v8-and-Easy-OCR\runs\detect\predict\crops\license-plate\temp.jpg'
            
            if os.path.exists(crop_license_image):
                st.image(bbox_image)
                st.success('Vehicle Plate Detected Successfully')
            else:
                st.error('No License Plate Detected')

            read = easyocr.Reader(['en'])
            result = read.readtext(crop_license_image, detail = 0)
            
            for text in result:
                st.subheader(f':red[Vehicle License Plate Number] : :green[{text}]')

            os.remove(os.path.join(temporary_image_path, 'temp.jpg'))
            shutil.rmtree(r'G:\GUVI_DATA_SCIENCE\Project\Vehicle-License-Plate-Recogition-using-YOLO-v8-and-Easy-OCR\runs\detect\predict')

    except:
        st.caption('An Error Occured')