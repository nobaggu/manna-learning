"""isRel — 검색된 문서가 질문과 관련 있는지 LLM에게 판단시키기"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def is_relevant(query: str, document: str) -> bool:
    """
    LLM에게 문서가 질문에 답하는 데 관련 있는지 yes/no로 판단시킴.

    Args:
        query: 사용자 질문
        document: 검색된 문서 텍스트

    Returns:
        True (관련 있음) 또는 False (무관)
    """
    prompt = f"""다음 문서가 아래 질문에 답하는 데 관련 있는지 판단해주세요.

[문서]
{document}

[질문]
{query}

답변은 반드시 'yes' 또는 'no'로만 시작하세요. 다른 설명 금지."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=10,
        temperature=0,  # 일관성 위해 0으로
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


# === 테스트: 관련/무관 케이스 5개 ===
test_cases = [
    # (질문, 문서, 예상_결과)
    (
        "신입 개발자 진로",
        "신입 개발자가 커리어를 시작할 때는 작은 회사에서 다양한 경험을 쌓는 것이 유리합니다.",
        True,  # 정확히 답하는 문서
    ),
    (
        "이직 타이밍",
        "이직 시점은 현재 회사에서 더 이상 배울 게 없다고 느낄 때가 적기입니다.",
        True,
    ),
    (
        "오늘 점심 메뉴",
        "사회적 자본이 부족한 사람에게 멘토를 매칭하려면 먼저 멘티의 목표와 제약을 구조화해야 합니다.",
        False,  # 완전 무관
    ),
    (
        "GitHub 학습 흔적",
        "벡터 데이터베이스는 의미 기반 검색을 가능하게 하는 핵심 인프라입니다.",
        False,  # 같은 IT 도메인이지만 답변과 무관
    ),
    (
        "사람 사이 정보 흐름",
        "사회연결망분석(SNA)은 사용자 간 정보 흐름을 정량적으로 측정할 수 있게 해줍니다.",
        True,  # 다른 단어, 같은 의미
    ),
]

print("=" * 80)
print("isRel 단위 테스트")
print("=" * 80)

correct = 0
for query, doc, expected in test_cases:
    result = is_relevant(query, doc)
    status = "✅" if result == expected else "❌"
    if result == expected:
        correct += 1

    print(f"\n{status} 질문: {query}")
    print(f"   문서: {doc[:50]}...")
    print(f"   예상: {expected} / 결과: {result}")

print(f"\n정확도: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")