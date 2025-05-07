import streamlit as st
import pandas as pd
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import fitz  # PyMuPDF for text extraction from PDFs
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load('resume_svm.pkl')
vectorizer = joblib.load('resume_vector.pkl')

# Initialize PorterStemmer and stop words
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def cleaning_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text

# Preprocessing function
def preprocessing(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# PDF text extraction using PyMuPDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Fallback: OCR extraction from PDF
def extract_text_with_ocr(file):
    images = convert_from_bytes(file.read())
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

# Label mapping
label_mapping = {
    0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain', 4: 'Business Analyst',
    5: 'Civil Engineer', 6: 'Data Science', 7: 'Database', 8: 'DevOps Engineer',
    9: 'DotNet Developer', 10: 'ETL Developer', 11: 'Electrical Engineering', 12: 'HR',
    13: 'Hadoop', 14: 'Health and fitness', 15: 'Java Developer', 16: 'Mechanical Engineer',
    17: 'Network Security Engineer', 18: 'Operations Manager', 19: 'PMO',
    20: 'Python Developer', 21: 'SAP Developer', 22: 'Sales', 23: 'Testing', 24: 'Web Designing'
}

# Streamlit App
st.title("📄 Resume Classification with SVM")
st.write("Upload a resume PDF to classify its job category.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        # Try regular text extraction
        re_text = extract_text_from_pdf(uploaded_file)

        # If nothing found, try OCR
        if not re_text.strip():
            st.warning("No text found using standard method. Trying OCR...")
            uploaded_file.seek(0)
            resume_text = extract_text_with_ocr(uploaded_file)

        if resume_text.strip():
            if st.button("Classify"):
                cleaned = cleaning_text(resume_text)
                preprocessed = preprocessing(cleaned)
                vector = vectorizer.transform([preprocessed]).toarray()
                prediction = model.predict(vector)[0]
                st.success(f"✅ This resume is classified as: **{label_mapping[prediction]}**")
        else:
            st.error("The PDF appears to be unreadable or empty.")
    except Exception as e:
        st.error(f"Failed to process the PDF: {e}")
else:
    st.info("Please upload a PDF file.")
