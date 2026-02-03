# 🧬 AI 기반 기대수명 예측 모델 및 시각화 시스템
> **국가별 보건 지표 데이터를 활용한 기대수명 분석 및 머신러닝 예측 솔루션**
> 
> 대규모 보건 데이터를 정밀 분석하여 기대수명에 영향을 미치는 핵심 인자를 규명하고, 최적화된 알고리즘을 통해 미래의 기대수명을 예측하는 웹 기반 시스템입니다.

---

## 📌 프로젝트 개요
* **진행 기간:** 2025.12.23 ~ 2025.12.30
* **핵심 목표:** * 다양한 보건 변수 간의 비선형적 관계 학습을 통한 정밀 예측
    * 국가별 보건 상황 정량적 평가 및 정책 수립을 위한 객관적 근거 제공
    * Streamlit을 활용한 사용자 친화적 예측 인터페이스 구축

## 🛠 기술 스택
- **Language:** ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- **Library:** - **Machine Learning:** Scikit-learn, XGBoost, LightGBM
  - **Data Analysis:** Pandas, NumPy
  - **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit (예측 웹 시스템 구현)

## ✨ 주요 기능 및 분석 내용
### 1. 데이터 전처리 및 탐색적 데이터 분석 (EDA)
- **결측치 처리:** 국가별/연도별 특성을 반영한 중앙값 대체법 활용
- **상관관계 분석:** Heatmap을 통해 기대수명과 상관관계가 높은 주요 변수(성인 사망률, 소득 지수, 교육 수준 등) 식별
- **이상치 제거:** IQR(Interquartile Range) 방식을 적용하여 데이터 품질 향상

### 2. 머신러닝 모델링
- **다양한 알고리즘 비교:** Linear Regression, Ridge, Lasso, Random Forest, XGBoost 모델 성능 비교
- **최적 모델 선정:** **Random Forest** (R² Score: 0.9619) 기반의 최종 모델 구축
- **변수 기여도 분석:** 분석 결과, 'HIV/AIDS' 지표가 전체 예측력의 58.1%를 차지하는 핵심 요인임을 증명



### 3. 예측 웹 서비스 구축
- 사용자가 직접 보건 지표를 입력하고 즉각적인 기대수명 예측 결과를 확인할 수 있는 대화형 UI 구현
- 예측값과 실제값의 비교 산점도 및 잔차 분석 그래프 제공

## 🏗 시스템 프로세스
1. **Raw Data**: 국가별 공공 보건 데이터셋 수집
2. **Preprocessing**: 데이터 정제 및 스케일링 (Standardization)
3. **Training**: 다양한 회귀 모델 학습 및 하이퍼파라미터 튜닝
4. **Evaluation**: RMSE, MAE, R² 지표를 통한 성능 검증 및 pkl 모델 저장
5. **Deployment**: Streamlit 기반 웹 인터페이스 배포



## 🚀 분석 결과 및 성과
- **높은 예측 정확도:** 테스트 데이터셋 기준 약 96%의 높은 설명력 확보
- **핵심 요인 규명:** 단순 통계 수치를 넘어 기대수명 결정 요인 기여도를 정량적으로 확인
- **안정성 검증:** 잔차 산점도(Residual Plot)가 특정 패턴 없이 고르게 분포됨을 통해 모델의 타당성 입증

---
**Author:** 이유진
