import streamlit as st
import pandas as pd
import numpy as np
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load BERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle cases where text extraction fails
    return text

# Function to clean text
def clean_text(text):
    return " ".join(text.split())  # Removes extra spaces, new lines

# Function to rank resumes using BERT similarity
def rank_resumes(job_description, resumes):
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_embeddings = model.encode(resumes, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(job_embedding, resume_embeddings)[0].cpu().numpy()
    return similarities

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Streamlit UI
st.title("🚀 AI Resume Screening & Ranking System")
st.markdown("This tool ranks resumes based on their similarity to the job description using **BERT-based NLP AI**.")

# Input for job description
st.header("📌 Job Description")
job_description = st.text_area("Enter the job description")

# File uploader for resumes
st.header("📂 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Button to start ranking
if st.button("🔍 Start Ranking Resumes"):
    if uploaded_files and job_description:
        st.header("📊 Ranking Resumes...")

        resumes = []
        resume_names = []
        
        # Extract and clean text from resumes
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            cleaned_text = clean_text(text)
            resumes.append(cleaned_text)
            resume_names.append(file.name)

        # Rank resumes using BERT
        scores = rank_resumes(job_description, resumes)

        # Create a DataFrame for results
        results = pd.DataFrame({"Resume": resume_names, "Similarity Score": scores})

        # Sort by similarity score (Descending)
        results = results.sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)

        # Add Rank Column
        results.insert(0, "Rank", ["Rank " + str(i+1) for i in range(len(results))])

        # Display results
        st.write("### 📈 Ranked Resumes")
        st.dataframe(results)

        # Allow downloading results as CSV
        csv = results.to_csv(index=False)
        st.download_button(label="📥 Download Ranking as CSV", data=csv, file_name="resume_rankings.csv", mime="text/csv")

        # Show word cloud for the top resume
        st.write("### ☁️ Word Cloud for Top Ranked Resume")
        top_resume_text = resumes[np.argmax(scores)]
        generate_wordcloud(top_resume_text)
    else:
        st.warning("⚠️ Please enter a job description and upload resumes before ranking.")