
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

# 모델 정의 (학습된 계수 사용)
model = LogisticRegression()
model.coef_ = np.array([[2.5, 1.8, 1.2]])  # 예시 계수 (출석에 긍정적인 영향)
model.intercept_ = np.array([-1.0])        # 예시 절편
model.classes_ = np.array([0, 1])          # 0: 결석, 1: 출석

# UI
st.title("🎓 출석 예측기 (최근 3회 출결 기반)")

st.markdown("최근 3회차 수업에 출석했는지 선택해주세요:")

col1, col2, col3 = st.columns(3)
att1 = col1.selectbox("이전 1회차", ["출석", "결석"])
att2 = col2.selectbox("이전 2회차", ["출석", "결석"])
att3 = col3.selectbox("이전 3회차", ["출석", "결석"])

# 출석=1, 결석=0 으로 변환
X_input = np.array([[int(att1 == "출석"), int(att2 == "출석"), int(att3 == "출석")]])

if st.button("출석 예측하기"):
    prob = model.predict_proba(X_input)[0][1]  # 출석 확률
    if prob >= 0.5:
        st.success(f"✅ 출석할 가능성이 높습니다! (출석 확률: {prob*100:.1f}%)")
    else:
        st.error(f"⚠️ 결석할 가능성이 높습니다. (출석 확률: {prob*100:.1f}%)")
