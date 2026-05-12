import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()#현재 위치에서 .env 파일을 찾아 안에 적힌 key=value들을 이 프로그램의 환경변수로 등록
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#open ai api와 통신할 수 있는 클라이언트 객체 생성

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