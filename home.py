import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# 모델 불러오기 (예시)
# model = torch.load('path/to/your_model.pth')
# model.eval()

# 이미지 전처리 함수 정의
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Streamlit 앱 설정
st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #FF0000;
        text-align: center;
        margin-bottom: 30px;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    .hateful {
        color: #FF0000;
    }
    .not-hateful {
        color: #2E8B57;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">Hateful Meme Classifier</p>', unsafe_allow_html=True)

# 예제 이미지 표시
st.write("### Example Predictions")
example1 = Image.open("/Users/hunjunsin/Desktop/python/Fundamental_CE/Project/ISSUES/resources/datasets/harmeme/img/covid_memes_9.png").resize((300, 300))  # 예시로 혐오성 콘텐츠가 있는 이미지 파일 경로
example2 = Image.open("/Users/hunjunsin/Desktop/python/Fundamental_CE/Project/ISSUES/resources/datasets/harmeme/img/covid_memes_20.png").resize((300, 300))  # 예시로 혐오성 콘텐츠가 없는 이미지 파일 경로

col1, col2 = st.columns(2)

# 첫 번째 예제
with col1:
    st.image(example1, caption="Example 1", use_column_width=True)
    st.markdown('<p class="result hateful">🔴 Predicted: Hateful 😡</p>', unsafe_allow_html=True)

# 두 번째 예제
with col2:
    st.image(example2, caption="Example 2", use_column_width=True)
    st.markdown('<p class="result not-hateful">🟢 Predicted: Not Hateful 😇</p>', unsafe_allow_html=True)

# 파일 업로드 섹션
st.write("### Upload Your Meme")
uploaded_image = st.file_uploader("Upload an image file (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # 업로드된 이미지를 화면에 표시
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # 이미지 전처리
    input_tensor = preprocess_image(image)
    
    # 모델 예측 수행
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    # 예측 결과 표시
    if predicted.item() == 1:  # 예: 1이 혐오 콘텐츠일 때
        st.markdown('<p class="result hateful">🔴 This meme contains hateful content 😡</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result not-hateful">🟢 This meme is not hateful 😇</p>', unsafe_allow_html=True)