import configparser
from huggingface_hub import InferenceClient
import streamlit as st
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from PIL import Image
import io

# Read the API key from the config file
config = configparser.ConfigParser()
config.read('config.ini')
hf_api = config['API']['huggingface_api']

generated_text = ""

# Image-To-Text API setup
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {hf_api}"}

def query_image(image_file):
    response = requests.post(
        API_URL,
        headers=headers,
        data=image_file
    )
    return response.json()

# Streamlit UI
st.title("Image Caption Generator")
st.write("Upload an image and get an AI-generated description!")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Add process button
    if st.button('Generate Caption'):
        with st.spinner('Analyzing image...'):
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Get prediction
            try:
                result = query_image(img_byte_arr)
                st.success("Caption Generated!")
                generated_text = result[0]['generated_text']
                st.write(generated_text)
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")

print(generated_text)

#TTS

# API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
# headers = {"Authorization": f"Bearer {hf_api}"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.content

# audio_bytes = query({
# 	"inputs": "The answer to the universe is 42",
# })
# # You can access the audio with IPython.display for example
# from IPython.display import Audio
# Audio(audio_bytes)