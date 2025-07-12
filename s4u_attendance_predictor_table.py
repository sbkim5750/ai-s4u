
import streamlit as st
import pandas as pd

# 예측 결과 파일 로딩 (로컬에서 사용 시 CSV 또는 Excel 로딩 필요)
@st.cache_data
def load_data():
    df = pd.read_excel("attendance_prediction_result.xlsx")
    return df

df = load_data()

st.title("📊 학생별 출결 예측 결과")

# 필터: 출석예측 기준
filter_option = st.selectbox("출석 예측 결과 필터", ["전체", "출석예상", "결석위험"])
if filter_option != "전체":
    df = df[df['출석예측'] == filter_option]

# 테이블 출력
st.dataframe(df[['ID', '과목명', '단원명', '회차코드', '이전1회출석', '이전2회출석', '이전3회출석', '출석예측', '출석확률']].reset_index(drop=True))

# 다운로드
st.download_button(
    label="📥 예측 결과 다운로드",
    data=df.to_csv(index=False).encode('utf-8-sig'),
    file_name="학생별_출결예측결과.csv",
    mime="text/csv"
)
