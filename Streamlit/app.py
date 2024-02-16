import streamlit as st
import pandas as pd
import openpyxl

import torch
from transformers import  BertForSequenceClassification, BertConfig, AutoTokenizer
import torch.nn.functional as F

# 모델 로드 함수 정의
def load_model(model_path):
    # 모델 구성을 로드 (필요한 경우)
    config = BertConfig.from_pretrained('skt/kobert-base-v1',
                                    hidden_dropout_prob=0.1,
                                    attention_probs_dropout_prob=0.1,
                                    num_labels=3)
    
    # 모델 인스턴스 생성
    model = BertForSequenceClassification(config=config)
    
    # 저장된 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    return model

# 모델과 토크나이저 경로
model_path = r"C:\Users\82106\OneDrive\바탕 화면\Streamlit\my_model.pth"

# 모델 로드
model = load_model(model_path)
# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')


def classify_text(text):
    # 입력된 텍스트를 토크나이저로 처리
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # 모델 예측
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 확률 계산 및 예측
    probabilities = F.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    return prediction, probabilities[0].tolist()


# streamlit 앱 설계
def main():
    st.title("무신사 스탠다드 리뷰 분류")
    col1, col2 = st.columns(2)

    with col1:
        user_input = st.text_area("리뷰를 입력하세요:")
        if st.button("분류하기"):
            prediction, probabilities = classify_text(user_input)
            if prediction == 0:
                st.write("Result: Negative")
            elif prediction == 1:
                st.write("Result: Positive")
            elif prediction == 2:
                st.write("Result: Neutral")
            st.write("[Probabilities]")
            st.write(f"Negative: {probabilities[0]:.4f}, Positive: {probabilities[1]:.4f}, Neutral: {probabilities[2]:.4f}")

    with col2:
        # Load and display Excel file
        excel_file = r'C:\Users\82106\OneDrive\바탕 화면\Streamlit\labeled_data.xlsx'
        labeled_df = pd.read_excel(excel_file)

        st.write('##### > 리뷰 현황')
        st.dataframe(labeled_df[0], width=500, height=300, hide_index=1) 

        st.bar_chart(data=labeled_df['classification'])
        


if __name__ == "__main__":
    main()
