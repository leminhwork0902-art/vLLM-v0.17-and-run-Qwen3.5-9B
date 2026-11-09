import os
from openai import OpenAI

cl = OpenAI(base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), api_key="EMPTY")


def ask(prompt, system="Bạn là trợ lý phân tích dữ liệu bán hàng DataMate.", thinking=False, max_tokens=2048):
    r = cl.chat.completions.create(
        model=os.environ.get("VLLM_MODEL", "qwen3.5-9b"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=max_tokens,
        # vLLM chi nhan enable_thinking qua chat_template_kwargs
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    return r.choices[0].message.content


if __name__ == "__main__":
    print(ask("Tháng 6/2025 cửa hàng bán được tổng cộng bao nhiêu tiền?"))
