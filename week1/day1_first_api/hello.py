import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("1. API 키 확인:", api_key[:15] + "..." if api_key else "NOT FOUND")

client = OpenAI(api_key=api_key)

print("2. OpenAI에 요청 보내는 중...")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "사회적 자본 비대칭이 뭔지 한 문단으로 설명해줘"}
    ]
)

print("3. 응답:")
print(response.choices[0].message.content)
print("4. 끝.")