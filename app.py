import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

# cv2 ko pehle force-import karo
try:
    import cv2
except ImportError:
    import types
    import sys
    cv2_mock = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2_mock

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from fpdf import FPDF
from datetime import datetime

# --- 1. ADVANCED RESPONSIVE CONFIG ---
st.set_page_config(
    page_title="Dental AI Report System",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Mobile Responsiveness & UI
st.markdown("""
    <style>
    /* Main container padding for mobile */
    .main > div {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Make images responsive */
    img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }
    /* Custom Card Style for Mobile */
    .stAlert {
        border-radius: 15px;
    }
    /* Full width buttons on mobile */
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #003366;
        color: white;
    }
    /* Header optimization */
    @media (max-width: 640px) {
        .main h1 {
            font-size: 1.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PDF CLASS SETUP (Unchanged) ---
class DentalPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, 'Dental OPG AI Diagnostic Report', 0, 1, 'C')
        self.ln(5)
        self.set_draw_color(0, 51, 102)
        self.line(10, 22, 200, 22)

# PDF Generation Function
def generate_pdf(original_img, result_img, findings, suggestions):
    pdf = DentalPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'R')
    
    original_img.save("temp_orig.jpg")
    cv2.imwrite("temp_res.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0)
    pdf.cell(95, 10, 'Original OPG X-ray', 0, 0)
    pdf.cell(95, 10, 'AI Detection Results', 0, 1)
    
    pdf.image("temp_orig.jpg", x=10, y=None, w=90)
    pdf.image("temp_res.jpg", x=105, y=pdf.get_y()-52, w=90)
    pdf.ln(60)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, ' Clinical Findings:', 1, 1, 'L', fill=True)
    pdf.set_font('Arial', '', 11)
    for find in findings:
        pdf.multi_cell(0, 8, f"- {find}")
    
    pdf.ln(5)
    pdf.set_fill_color(230, 242, 255)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, ' AI-Based Recommendations:', 1, 1, 'L', fill=True)
    pdf.set_font('Arial', 'I', 11)
    pdf.multi_cell(0, 8, suggestions)
    
    pdf.set_y(-30)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(150)
    pdf.multi_cell(0, 5, "Disclaimer: It is NOT a final diagnosis. Please consult a certified Dentist for clinical evaluation.", 0, 'C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 3. APP UI ---
st.title("🦷 Dental OPG AI Analysis & Report System")
st.write("Professional X-ray analysis with automated patient suggestions.")

# Model Load
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Sidebar (Better for mobile to keep controls tucked away)
st.sidebar.header("🎛️ Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.40)

# File uploader (Mobile optimized)
uploaded_file = st.file_uploader("Upload OPG X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Use columns that wrap on mobile automatically
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("🖼️ Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.info("🔍 AI Analysis")
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot()
        st.image(res_plotted, use_container_width=True)

    # Findings logic
    findings = []
    detected_classes = set()
    for box in results[0].boxes:
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        findings.append(f"{label} detected with {conf:.2%} confidence")
        detected_classes.add(label)

    suggestion_map = {
        "Caries": "Dental filling is recommended. Use fluoride toothpaste and limit sugary intake.",
        "Infection": "Possible root canal or antibiotic therapy required. Urgent clinical checkup advised.",
        "Fractured Teeth": "Tooth stabilization or crowning might be needed. Avoid chewing from this side.",
        "Impacted teeth": "Surgical evaluation for extraction is advised if pain or crowding occurs.",
        "BDC-BDR": "Severe decay detected. Deep scaling or extraction assessment needed.",
        "Healthy Teeth": "No immediate issues found. Continue regular 6-month dental checkups."
    }

    st.divider()
    st.subheader("📋 Detection Summary & Advice")
    
    full_suggestions = ""
    if detected_classes:
        # Professional cards for findings
        for item in detected_classes:
            advice = suggestion_map.get(item, "Consult dentist for specific treatment.")
            st.warning(f"**{item}**: {advice}")
            full_suggestions += f"{item}: {advice}\n"
    else:
        st.success("✨ Everything looks healthy! Keep up the good oral hygiene.")
        full_suggestions = "No major issues detected. Maintain regular checkups."

    # PDF Download Section (Responsive Button)
    st.divider()
    if st.button("📄 Prepare PDF Report"):
        with st.spinner('Generating professional report...'):
            pdf_bytes = generate_pdf(image, res_plotted, findings, full_suggestions)
            st.download_button(
                label="📥 Download Full Medical Report",
                data=pdf_bytes,
                file_name=f"Dental_Analysis_{datetime.now().strftime('%d%m%Y')}.pdf",
                mime="application/pdf"
            )