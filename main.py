import configparser
from huggingface_hub import InferenceClient
import streamlit as st
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from PIL import Image
import io

# Config setup
config = configparser.ConfigParser()
config.read('config.ini')
hf_api = config['API']['huggingface_api']

# API setup
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {hf_api}"}

def query_image(image_file):
    response = requests.post(API_URL, headers=headers, data=image_file)
    return response.json()

def generate_story(text):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    from langchain_core.messages import (
        HumanMessage,
        SystemMessage,
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    messages = [
        SystemMessage(content="You are a story teller, if you recieved a message, you should respond to it with story based on that phrase or sentence. You're story have unespected plot. The words are no more longer than 100 words."),
        HumanMessage(content=text),
    ]
    
    return chat_model.invoke(messages).content

# Streamlit UI
st.title("Image to Story Generator")
st.write("Upload an image and get an AI-generated story based on it!")

# Sidebar for image upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main content
if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Process button
    if st.button('Generate Story'):
        with st.spinner('Processing generating story...'):
            try:
                # Generate caption
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()
                
                caption_result = query_image(img_byte_arr)
                caption = caption_result[0]['generated_text']
                
                # Generate story
                story = generate_story(caption)
                
                st.subheader("Generated Story")
                st.write(story)
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
else:
    st.info("Please upload an image to get started!")


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