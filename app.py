import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoImageProcessor,
    ViTForImageClassification
)
from PIL import Image
import base64
import warnings
warnings.filterwarnings("ignore")

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Medi-Co", page_icon="2.png", layout="wide")

# ---------------------- Custom CSS ----------------------
st.markdown("""
    <style>
        body {
            background-color: #f4f6fb;
            color: #2c2c2c;
        }

        .main-title {
            text-align: center;
            font-size: 2.5em;
            font-weight: 800;
            color: #2b6cb0;
            margin-bottom: 0.2em;
        }

        .subtitle {
            text-align: center;
            font-size: 1.1em;
            color: #555;
            margin-bottom: 25px;
        }

        .section-header {
            font-size: 1.3em;
            font-weight: 700;
            color: #264653;
            margin-bottom: 8px;
        }

        /* Response Box (white card style) */
        .response-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #d0d7de;
            font-size: 1.05em;
            color: #1a202c;
            line-height: 1.6;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            animation: fadeIn 1.2s ease-in-out;
        }

        .info-box {
            background-color: #e8f3ff;
            padding: 15px;
            border-radius: 10px;
            font-size: 0.95em;
            color: #03396c;
            border-left: 5px solid #3a86ff;
        }

        .warn-box {
            background-color: #fff4e6;
            padding: 15px;
            border-radius: 10px;
            font-size: 0.95em;
            color: #663c00;
            border-left: 5px solid #ffa600;
        }

        .footer {
            text-align: center;
            color: #808080;
            margin-top: 50px;
            font-size: 0.9em;
        }
        .footer a {
            color: #3a86ff;
            text-decoration: none;
        }

        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .fade-in {animation: fadeIn 1.2s ease-in-out;}
    </style>
""", unsafe_allow_html=True)


# ---------------------- Header with Centered Logo ----------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    logo_base64 = get_base64_image("2.png")
    st.markdown(f"""
        <div class='centered fade-in'>
            <img src='data:image/png;base64,{logo_base64}' width='110' style='margin-bottom:10px;'>
        </div>
        <h1 class='main-title fade-in'>Medi-Co — Your AI Health Assistant</h1>
        <p class='subtitle fade-in'>Ask medical questions or upload an image for analysis 🧠🩺</p>
    """, unsafe_allow_html=True)
except Exception:
    st.warning("⚠️ Logo not found — please check path '2.png'")

# ---------------------- Load Models ----------------------
@st.cache_resource
def load_models():
    text_model_name = "microsoft/phi-2"
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    text_model = AutoModelForCausalLM.from_pretrained(
        text_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    text_model.config.pad_token_id = text_model.config.eos_token_id

    image_model_name = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(image_model_name)
    image_model = ViTForImageClassification.from_pretrained(image_model_name)

    return text_model, text_tokenizer, image_model, image_processor

text_model, text_tokenizer, image_model, image_processor = load_models()

# ---------------------- Helper Functions ----------------------
def generate_text_response(prompt: str) -> str:
    inputs = text_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = text_model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_k=50,
        temperature=0.8,
        repetition_penalty=1.2,
        pad_token_id=text_tokenizer.eos_token_id,
    )
    response = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    return response.strip()

def analyze_image(image: Image.Image) -> str:
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = image_model(**inputs)
    pred_id = outputs.logits.argmax(-1).item()
    return image_model.config.id2label[pred_id]

def ai_assistant(user_text=None, image=None):
    context = ""
    if image:
        img_label = analyze_image(image)
        context += f"The uploaded image seems to contain: {img_label}. "
    if user_text:
        context += f"User says: {user_text}\n"

    full_prompt = (
    "You are Medi-Co, a professional yet friendly AI health assistant.\n"
    "Your goal is to help users understand their symptoms in simple terms, "
    "provide possible causes, and suggest safe home remedies. "
    "Do NOT prescribe medication or make definitive diagnoses.\n\n"
    f"User Query: {context}\n\n"
    "Medi-Co Response:"
    )

    response = generate_text_response(full_prompt)
    return response if response else "I'm sorry, I couldn’t generate a helpful response this time."

# ---------------------- Layout ----------------------
st.divider()
col1, col2 = st.columns([1.6, 1], gap="large")

with col1:
    st.markdown("<p class='section-header'>💬 Ask Medi-Co</p>", unsafe_allow_html=True)
    user_query = st.text_area("Your Question:", placeholder="e.g., I have a cough, what should I do?")
    submitted = st.button("🔍 Ask Medi-Co", use_container_width=True)

    if submitted and (user_query):
        with st.spinner("🧠 Medi-Co is thinking..."):
            answer = ai_assistant(user_text=user_query)
        st.markdown("<p class='section-header'>🩺 Medi-Co’s Response</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)
    elif submitted:
        st.warning("Please enter a question before submitting.")

with col2:
    st.markdown("<p class='section-header'>📸 Upload Image (Optional)</p>", unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)
        if st.button("🧠 Analyze Image", use_container_width=True):
            with st.spinner("🔍 Analyzing image..."):
                label = analyze_image(image)
            st.success(f"🩻 Prediction: **{label}**")
    st.divider()
    st.markdown("<div class='info-box'>🧠 <b>How it Works:</b><br>Medi-Co uses <b>Phi-2</b> for text and <b>ViT</b> for medical image analysis. It provides educational, AI-based insights.</div>", unsafe_allow_html=True)
    st.markdown("<div class='warn-box'>⚠️ <b>Disclaimer:</b><br>Medi-Co is not a substitute for professional medical advice. Always consult a licensed doctor for accurate diagnosis or treatment.</div>", unsafe_allow_html=True)

# ---------------------- Footer ----------------------
st.markdown("<div class='footer'>💻 Created by <b>Anuj</b> & <b>Varun</b> | Powered by <a href='https://huggingface.co' target='_blank'>Hugging Face Transformers</a></div>", unsafe_allow_html=True)
