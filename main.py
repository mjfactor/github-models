import configparser
from huggingface_hub import InferenceClient
import streamlit as st
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from PIL import Image
from IPython.display import Audio
import io

# Fetching Hugging Face API key from config.ini
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

TTS_API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
headers = {"Authorization": f"Bearer {hf_api}"}

def generate_speech(text):
    response = requests.post(TTS_API_URL, headers=headers, json={"inputs": text})
    return response.content


# Streamlit UI
st.title("Image to Story Generator")
st.write("Upload an image and get an AI-generated story based on it!")

# Sidebar for image upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main content
if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Generate Caption, Story and Audio'):
        with st.spinner('Processing...'):
            try:
                # Generate caption
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()
                
                caption_result = query_image(img_byte_arr)
                caption = caption_result[0]['generated_text']
                
                with col2:
                    st.subheader("Image Caption")
                    st.write(caption)
                
                # Generate story
                story = generate_story(caption)
                
                st.subheader("Generated Story")
                st.write(story)
                
                # Generate audio
                with st.spinner('Generating audio...'):
                    audio_bytes = generate_speech(story)
                    st.subheader("Listen to the Story")
                    st.audio(audio_bytes, format='audio/mp3')
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
else:
    st.info("Please upload an image to get started!")