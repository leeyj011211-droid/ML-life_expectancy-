import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 모델 및 변수 리스트 불러오기
with open("life_expectancy_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
features = data["features"]

# 웹 페이지 제목 및 설명
st.set_page_config(page_title="기대수명 예측 서비스", layout="wide")
st.title("🌍 국가별 보건 지표 기반 기대수명 예측")
st.markdown("""
이 서비스는 국가의 보건 및 경제 지표를 입력하여 해당 국가의 **예상 기대수명**을 AI 모델로 예측합니다.
수치를 조정해 보세요.
""")

st.divider()

# 2. 사용자 입력 받기
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 경제 및 교육 지표")
    income_comp = st.slider("자원 배분의 소득 구성 지수 (Income composition)", 
                            min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    schooling = st.slider("평균 교육 연수 (Schooling)", 
                          min_value=0.0, max_value=20.7, value=10.0, step=0.1)
    under_five = st.number_input("5세 미만 사망자 수 (under-five deaths, 명)", 
                                 min_value=0, max_value=2500, value=42)
    five_death_log = np.log1p(under_five) 

    adult_mortality = st.number_input("성인 사망률 (Adult Mortality, 1,000명당)", 
                                      min_value=1, max_value=723, value=160, step=1)
        
with col2:
    st.subheader("🏥 보건 및 신체 지표")
    thinness = st.slider("10대 저체중 유병률 (thinness 1-19 years, %)", 
                         min_value=0.0, max_value=27.7, value=4.0, step=0.1)
    hiv_raw = st.number_input("HIV/AIDS 유병률 (0-50.6 사이 입력)", 
                              min_value=0.0, max_value=50.6, value=0.1, step=0.1)
    hiv_log = np.log1p(hiv_raw) 
    bmi = st.number_input("평균 BMI (BMI)", 
                          min_value=0.0, max_value=87.0, value=38.0, step=0.1)

st.divider()

# 3. 예측 실행 버튼
if st.button("기대수명 예측하기", use_container_width=True):
    input_data = {
        'Income composition of resources': income_comp,
        'HIV_log': hiv_log,
        ' BMI ': bmi,
        'Adult Mortality': adult_mortality,
        'five deaths_log': five_death_log,
        'Schooling': schooling,
        ' thinness  1-19 years': thinness
    }
    
    input_df = pd.DataFrame([input_data])[features]
    prediction = model.predict(input_df)
    
    st.markdown(f"""
    <div style="text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #0e1117;">예상 기대수명</h2>
        <h1 style="color: #ff4b4b;">{prediction[0]:.2f} 세</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # 그래프 설정
    st.subheader("💡 변수 중요도")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette="pastel", ax=ax)
    
    # 그래프 테두리 제거 및 디자인 깔끔하게 정리
    ax.set_title("Key Indicators for Prediction (Sorted by Importance)")
    sns.despine() # 위쪽과 오른쪽 테두리 제거
    
    st.pyplot(fig)