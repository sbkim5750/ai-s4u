from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd

# 키워드 리스트
KEYWORDS = ["학습", "심리", "건강", "진로", "생활", "성적", "기타"]

# 벡터화 준비
vectorizer = TfidfVectorizer()
keyword_vec = vectorizer.fit_transform(KEYWORDS)

# Streamlit UI 설정
st.set_page_config(page_title="S4U 키워드 자동 분류기", layout="wide")
st.title("🔍 상담내용 기반 키워드 자동 분류기")

st.markdown("**📌 상담내용을 입력하고 `키워드 추출` 버튼을 눌러주세요.**")
text = st.text_area("✏️ 상담 내용을 입력하세요", height=200)

if st.button("🔍 키워드 추출") and text:
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, keyword_vec).flatten()
    keyword_matches = sorted(zip(KEYWORDS, sims), key=lambda x: x[1], reverse=True)

    # 상위 3개 키워드 추출 (기타 포함될 수 있음)
    top_keywords = [kw for kw, score in keyword_matches[:3]]

    st.success("😎예측된 키워드 Top 3:")
    for i, kw in enumerate(top_keywords, 1):
        st.markdown(f"- {i}. **{kw}**")

# 사이드바 안내
with st.sidebar:
    st.header("ℹ️ 분류 기준 안내")
    st.markdown("- 키워드 후보: `학습`, `심리`, `건강`, `진로`, `생활`, `기타`")
    st.markdown("- TF-IDF + Cosine Similarity 기반")
    st.markdown("- 가장 유사한 키워드 3개를 추천합니다.")

# 하단 배너
st.markdown("---")
st.caption("ⓒ 2025 S4U Keyword Extractor")
