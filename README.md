# AI Resume Screening & Ranking System

This project is a **Streamlit-based web application** that ranks resumes based on their similarity to a given job description using **AI-powered NLP techniques**.
Try the application: https://resume-rank.streamlit.app/

## 🚀 Features
- **Upload multiple resumes (PDF format)**
- **Enter a job description**
- **AI-based ranking using NLP (TF-IDF & Cosine Similarity)**
- **Download ranking results as CSV**
- **Visualize top-ranked resume with a word cloud**

---

## 🛠️ Technologies Used
- **Python**
- **Streamlit** (UI framework)
- **Scikit-learn** (TF-IDF & Cosine Similarity)
- **PyPDF2** (Extract text from PDFs)
- **Pandas** (Data handling)
- **NumPy** (Numerical computations)
- **Wordcloud & Matplotlib** (Visualization)
- **Torch & Sentence Transformers** (For future AI model integration)

---

## 📥 Installation & Setup

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/rajesh31082001/Resume_Ranking.git
cd Resume_Ranking
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the App**
```bash
streamlit run resume_ranking.py
```

---

## 📌 How to Use?
1. **Enter** the job description in the text box.
2. **Upload** multiple PDF resumes.
3. Click the **Start Ranking Resumes** button.
4. View the ranked resumes with a **score & rank**.
5. Download the rankings as a **CSV file**.
6. View a **word cloud** of the top-ranked resume.

---

## 🌍 Deploying on Streamlit Cloud
1. **Push your project to GitHub**
2. **Go to** [Streamlit Cloud](https://share.streamlit.io/)
3. **Deploy with:**
   - **Repository:** `your-username/resume-ranking-app`
   - **Branch:** `main`
   - **Main file path:** `resume_ranking.py`
4. **Click Deploy!** 🚀

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repo and submit a PR.

---

## 📧 Contact
For any issues, feel free to contact **Rajesh Pal** via GitHub or email.

📌 Happy Coding! 🎉

