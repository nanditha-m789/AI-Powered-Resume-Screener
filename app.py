import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def screen_resumes(job_description, resume_folder="resumes"):
    if not os.path.exists(resume_folder):
        print(f"Folder '{resume_folder}' not found. Please create it and add PDF resumes.")
        return

    resumes = []
    resume_names = []

    for file in os.listdir(resume_folder):
        if file.endswith(".pdf"):
            path = os.path.join(resume_folder, file)
            text = extract_text_from_pdf(path)
            if text:
                resumes.append(text)
                resume_names.append(file)

    if not resumes:
        print("No resumes found in folder.")
        return

    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked = sorted(zip(resume_names, similarity), key=lambda x: x[1], reverse=True)

    print("\n--- RESUME RANKING ---\n")
    for name, score in ranked:
        print(f"{name} - Match: {score*100:.2f}%")

job_desc = "Looking for Python Developer with skills in Machine Learning, NLP, AI, Data Science, Flask."

if __name__ == "__main__":
    screen_resumes(job_desc)
