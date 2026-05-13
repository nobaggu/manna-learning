"""isSup — LLM 답변이 제공 문서로 뒷받침되는지 검증"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def is_supported(answer: str, documents: list[str]) -> bool:
    """
    LLM 답변이 제공된 문서들에서 *직접 또는 합리적으로 추론 가능*한지 판단.

    Returns:
        True: 모든 주장이 문서로 뒷받침됨
        False: 문서에 없는 정보가 포함되거나 문서를 왜곡함
    """
    docs_text = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])

    prompt = f"""다음 답변이 아래 자료들로 *뒷받침*되는지 판단해주세요.

[제공된 자료]
{docs_text}

[답변]
{answer}

판단 기준:
1. 답변의 모든 주장이 자료에 *직접 명시*되었거나 *합리적으로 추론* 가능하면 'yes'
2. 답변에 자료에 *없는 새로운 사실·숫자·인용*이 포함되면 'no'
3. 답변이 자료의 내용을 *반대로 말하거나 왜곡*하면 'no'

답변은 반드시 'yes' 또는 'no'로만 시작하세요. 다른 설명은 그 뒤에 한 문장 이내로."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=80,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content.strip()
    verdict = output.lower().startswith("yes")
    return verdict, output


# === 테스트 케이스 ===
# 동일한 자료를 두고, 답변만 4가지 버전으로 만들어서 isSup 정확도 확인
documents = [
    "신입 개발자가 커리어를 시작할 때는 작은 회사에서 다양한 경험을 쌓는 것이 유리합니다.",
    "이직 시점은 현재 회사에서 더 이상 배울 게 없다고 느낄 때가 적기입니다.",
]

test_answers = [
    # (라벨, 답변, 예상_isSup)
    (
        "충실",
        "신입 개발자는 작은 회사에서 다양한 경험을 쌓는 것이 유리하며, 더 이상 배울 게 없다고 느낄 때 이직을 고민하는 것이 좋습니다.",
        True,  # 자료 그대로
    ),
    (
        "외부지식혼입",
        "신입 개발자는 보통 연봉 3000만원대로 시작하며, 3년 차에 평균 20% 인상되어 이직하는 게 일반적입니다.",
        False,  # 연봉·년차 정보는 자료에 없음
    ),
    (
        "왜곡",
        "신입 개발자는 큰 회사에서 시작하는 것이 가장 좋으며, 현재 회사에서 잘 배우고 있어도 적극적으로 이직을 시도해야 합니다.",
        False,  # 자료와 정반대
    ),
    (
        "과도한 추론",
        "신입 개발자는 작은 회사에서 다양한 경험을 쌓는 것이 유리하므로, 대기업 입사는 결국 후회로 이어집니다.",
        False,  # "유리하다" → "대기업은 후회" 는 비약
    ),
    (
        "부분 혼합",
        "신입 개발자는 작은 회사에서 다양한 경험을 쌓는 것이 유리하며, 특히 시리즈 A 단계의 스타트업이 가장 좋은 선택입니다.",
        False,  # "시리즈 A" 는 자료에 없음
    ),
]

print("=" * 80)
print("isSup 단위 테스트")
print("=" * 80)

correct = 0
for label, answer, expected in test_answers:
    verdict, raw_output = is_supported(answer, documents)
    status = "✅" if verdict == expected else "❌"
    if verdict == expected:
        correct += 1

    print(f"\n{status} [{label}] 예상: {expected} / 결과: {verdict}")
    print(f"   답변: {answer[:60]}...")
    print(f"   LLM 응답: {raw_output[:80]}")

print(f"\n정확도: {correct}/{len(test_answers)} ({correct/len(test_answers)*100:.0f}%)")