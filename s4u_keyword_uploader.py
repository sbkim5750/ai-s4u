
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 키워드 사전
KEYWORDS = ['학습', '심리', '건강', '출결', '가정', '진로', '기타']

st.title("📎 상담내용 엑셀 업로드 → 키워드 자동 추출기")

# 파일 업로드
uploaded_file = st.file_uploader("📂 상담내용 엑셀 파일을 업로드하세요", type=["xlsx"])

if uploaded_file:
    # 데이터 로드
    try:
        df = pd.read_excel(uploaded_file)
        if '상담내용' not in df.columns:
            st.error("❗ '상담내용' 컬럼이 존재하지 않습니다.")
        else:
            df['상담내용'] = df['상담내용'].fillna('').astype(str)

            # TF-IDF 학습
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
                return ", ".join(matched_keywords)

            # 키워드 추출 적용
            df['추출 키워드'] = df['상담내용'].apply(extract_keywords)

            # 결과 출력
            st.success("✅ 키워드 추출 결과:")
            st.dataframe(df[['상담내용', '추출 키워드']].head(50))

            # 다운로드 버튼
            output_file = "keyword_extracted_result.xlsx"
            df.to_excel(output_file, index=False)
            with open(output_file, "rb") as f:
                st.download_button(
                    label="📥 결과 엑셀 다운로드",
                    data=f,
                    file_name="상담_키워드_추출결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
