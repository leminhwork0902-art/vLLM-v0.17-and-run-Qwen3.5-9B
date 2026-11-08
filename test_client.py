import os
from openai import OpenAI

# Endpoint vLLM do Minh dung. Doi qua bien moi truong neu can.
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.5-9b")

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def query_qwen(
    prompt,
    system_prompt="Bạn là trợ lý phân tích dữ liệu bán hàng DataMate.",
    thinking=False,
    max_tokens=2048,
):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,  # greedy
        max_tokens=max_tokens,
        # Dung cach nay: vLLM chi nhan enable_thinking qua chat_template_kwargs.
        # De enable_thinking o cap tren cung se bi bo qua (thinking van bat).
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Sending request to vLLM server...")
    print("Result:\n", query_qwen("Tháng 6/2025 cửa hàng bán được tổng cộng bao nhiêu tiền?"))
