# 📄 Resume Classification using Machine Learning

This project classifies resumes into predefined job categories using NLP techniques and displays results through a **Streamlit** interface.

## 🚀 Features
- Text extraction and preprocessing from resumes (PDF)
- TF-IDF vectorization
- Machine Learning classification (SVM)
- Interactive Streamlit UI with support for scanned PDFs using OCR

## 🧠 Machine Learning
- **Model Used**: Support Vector Machine (SVM)
- **Vectorization**: TF-IDF
- **Categories**: Data Science, Web Developer, HR, etc.
- **Dataset**: Based on labeled resume samples

## 📦 Technologies
- Python
- Scikit-learn
- Pandas
- Numpy
- Streamlit
- PyMuPDF (for text-based PDFs)
- pytesseract + pdf2image (for scanned PDFs)

## 📷 Demo
![{1EB65C7E-D5BB-438E-AC27-BA905F2282B1}](https://github.com/user-attachments/assets/60eb32a4-5d53-4a1a-b6b9-cd78215eac2a)
 <!-- Add a screenshot in a 'screenshots' folder -->

## ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-classification-streamlit.git
cd resume-classification-streamlit

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
