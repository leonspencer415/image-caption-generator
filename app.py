import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    return processor, model

st.set_page_config(page_title="BLIP Large Captioning", layout="wide")

st.title("WAN 2.2 Dataset Caption Generator (BLIP Large)")
st.write("Upload an image and generate clean BLIP-Large captions for WAN datasets.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Image Preview", use_column_width=True)

    with st.spinner("Generating caption..."):
        processor, model = load_model()
        inputs = processor(image, return_tensors="pt")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
        )
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

    st.success("Caption generated:")
    st.write(f"**{caption}**")
