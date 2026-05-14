"""isUse — 답변이 사용자에게 실제로 유용한지 검증"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def is_useful(query: str, answer: str) -> tuple[bool, str]:
    """
    답변이 사용자 질문에 실질적으로 유용한지 판단.

    Returns:
        (verdict: bool, reason: str)
    """
    prompt = f"""다음 답변이 사용자의 질문에 *실질적으로 유용*한지 판단해주세요.

[질문]
{query}

[답변]
{answer}

판단 기준:
1. 답변이 질문에 *직접* 답하는가? (동문서답 X)
2. 답변에 *구체적이고 실행 가능한* 정보가 담겨 있는가? (추상적 동어반복 X)
3. 사용자가 이 답변을 읽고 *다음에 무엇을 할지* 판단할 수 있는가?
4. "사람마다 다르니 본인 판단" 같은 *회피성 답변*은 아닌가?

답변은 'yes' 또는 'no'로 시작, 그 뒤에 한 문장 이내로 이유를 설명하세요."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=80,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content.strip()
    return output.lower().startswith("yes"), output


# === 테스트: 같은 질문에 6가지 답변 품질 ===
query = "이직 언제 해야 하나요?"

test_answers = [
    # (라벨, 답변, 예상_isUse)
    (
        "구체적·실행가능",
        "현재 회사에서 더 이상 배울 게 없다고 느낄 때가 이직 적기입니다 [1].",
        True,
    ),
    (
        "추상적 동어반복",
        "이직은 신중하게 결정해야 하는 중요한 일입니다.",
        False,  # 사실이지만 쓸모없음
    ),
    (
        "회피성 답변",
        "사람마다 상황이 다르므로 본인의 판단이 가장 중요합니다.",
        False,  # 책임 회피
    ),
    (
        "동문서답",
        "신입 개발자는 작은 회사에서 다양한 경험을 쌓는 게 좋습니다.",
        False,  # 질문과 다른 답
    ),
    (
        "부적절한 거절",
        "이직 시점에 대한 정보가 충분하지 않습니다.",
        False,  # 실제로는 답이 있는데 약하게 답
    ),
    (
        "충분히 상세",
        "현재 회사에서 더 이상 배울 게 없다고 느낄 때가 이직 적기입니다. 본인의 성장 둔화 신호(반복 업무, 도전 부족 등)를 점검해보세요.",
        True,
    ),
]

print("=" * 80)
print(f"질문: {query}")
print("=" * 80)

correct = 0
for label, answer, expected in test_answers:
    verdict, raw = is_useful(query, answer)
    status = "✅" if verdict == expected else "❌"
    if verdict == expected:
        correct += 1

    print(f"\n{status} [{label}] 예상: {expected} / 결과: {verdict}")
    print(f"   답변: {answer}")
    print(f"   판단 이유: {raw[:80]}")

print(f"\n정확도: {correct}/{len(test_answers)} ({correct/len(test_answers)*100:.0f}%)")