import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("4-1. 2025_0711_s4u_상담데이터.xlsx", sheet_name="Result 1")
    df['상담내용'] = df['상담내용'].fillna('').astype(str)
    return df

df = load_data()

# TF-IDF 벡터라이저 학습
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['상담내용'])

# 유사 상담 찾기 함수
def find_similar_cases(user_input, top_n=5):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[::-1][:top_n]
    return df.iloc[similar_indices][['마스킹이름', '상담내용', '상담날짜']]

# UI 구성
st.title("📋 상담내용 기반 유사 상담 검색기")

user_input = st.text_area("상담 내용을 입력하세요:", height=150)

if st.button("유사 상담 찾기"):
    if user_input.strip() == "":
        st.warning("상담 내용을 입력해주세요.")
    else:
        result = find_similar_cases(user_input)
        st.success("유사 상담 결과:")
        st.dataframe(result.reset_index(drop=True))
