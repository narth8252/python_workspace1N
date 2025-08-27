from transformers import pipeline

# 텍스트 생성 모델 로드
generator = pipeline("text-generation", model="gpt2")

# 텍스트 생성
generated_text = generator("Hugging Face is", max_length=50, num_return_sequences=1)
print(generated_text)
