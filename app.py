import os
import streamlit as st
from openai import OpenAI, OpenAIError
import requests
from rembg import remove, new_session
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key= os.environ.get("OPENAI_API_KEY"))

st.title("AI Background Replacement")

uploaded_image = st.file_uploader(
    "Upload an image (png/jpg) from which you want to remove the background:",
    type=["png", "jpg", "jpeg"]
)

prompt = st.text_input(
    "Enter a text prompt for your new background (e.g. 'A futuristic city at night'):",
    value="A serene beach at sunset"
)

if st.button("Execute"):
    if uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        with st.spinner("Processing..."):
            input_img = Image.open(uploaded_image)
            # session = new_session("u2net_cloth_seg")
            subject = remove(input_img)
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
            except OpenAIError as e:
                st.error(f"OpenAI API error: {e}")
                st.stop()

            img_url = response.data[0].url

            # Download the image
            try:
                bg_image_response = requests.get(img_url)
                bg_image = Image.open(BytesIO(bg_image_response.content)).convert("RGBA")
            except Exception as e:
                st.error(f"Failed to download generated image: {e}")
                st.stop()

            subject_resized = subject.resize(bg_image.size, Image.LANCZOS)

            final_img = bg_image.copy()
            final_img.paste(subject_resized, (0, 0), subject_resized)

            final_rgb = final_img.convert("RGB")

            st.image(final_rgb, caption="Final Image", use_container_width=True)

            buf = BytesIO()
            final_rgb.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="result.jpg",
                mime="image/jpeg"
            )
