# TensorFlow Lite 모델 불러오기 + 이미지 분류 코드 예시
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. 모델 경로
MODEL_PATH = r"C:\Users\Admin\Downloads\tm-my-image-model\model.tflite"

# 2. TFLite 인터프리터 초기화
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# 3. 입력, 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4. 이미지 전처리 함수
def preprocess_image(image_path, input_shape):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.float32)
    # 모델에 따라 정규화 필요하면 여기서 수행 (예: /255.0)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# 5. 이미지 경로 지정
image_path = r"C:\Users\Admin\Downloads\tm-my-image-model\test_image.jpg"

# 6. 전처리
input_shape = input_details[0]['shape']
input_data = preprocess_image(image_path, input_shape)

# 7. 입력 텐서에 데이터 세팅
interpreter.set_tensor(input_details[0]['index'], input_data)

# 8. 모델 실행
interpreter.invoke()

# 9. 출력 받아오기
output_data = interpreter.get_tensor(output_details[0]['index'])

# 10. 결과 확인 (예: softmax 확률 or 클래스 인덱스)
predicted_class = np.argmax(output_data)
print(f"Predicted class index: {predicted_class}")
print(f"Raw output: {output_data}")
