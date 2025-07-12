
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 키워드 사전
KEYWORDS = ['학습', '심리', '건강', '출결', '가정', '진로', '기타']

# 샘플 상담내용 데이터 (TF-IDF 학습용)
@st.cache_data
def load_sample_data():
    df = pd.read_excel("4-1. 2025_0711_s4u_상담데이터.xlsx", sheet_name="Result 1")
    df['상담내용'] = df['상담내용'].fillna('').astype(str)
    return df

df = load_sample_data()

# TF-IDF 벡터라이저 학습
vectorizer = TfidfVectorizer(max_features=1000)
vectorizer.fit(df['상담내용'])

# 키워드 추출 함수
def extract_keywords(text, top_n=10):
    tfidf_vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = tfidf_vec.toarray().flatten().argsort()[::-1]
    top_words = [feature_names[i] for i in sorted_indices[:top_n]]
    matched_keywords = [kw for kw in KEYWORDS if kw in top_words]
    if not matched_keywords:
        matched_keywords = ['기타']
    return matched_keywords

# Streamlit UI
st.title("📌 상담내용 키워드 자동 추출기")

user_input = st.text_area("상담 내용을 입력하세요:", height=150)

if st.button("키워드 추출"):
    if user_input.strip() == "":
        st.warning("상담 내용을 입력해주세요.")
    else:
        keywords = extract_keywords(user_input)
        st.success("추출된 키워드:")
        st.write(", ".join(keywords))
