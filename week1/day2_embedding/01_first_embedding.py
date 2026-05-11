import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "사회적 자본 비대칭"

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

vector = response.data[0].embedding

print(f"원본 텍스트: {text}")
print(f"벡터 차원 수: {len(vector)}")
print(f"앞 10개 값: {vector[:10]}")
print(f"뒤 5개 값: {vector[-5:]}")