"""
run_eval.py - SKELETON. Cac cho danh dau [ASSUMPTION] can sua theo format that
cua BTC khi co data/rulebook/README mau.

Luong: doc cau hoi trong /data -> agent (Qwen3.5-9B viet code pandas) -> chay code
-> ghi submission.jsonl + log tung cau. Cau loi/khong chac => de trong (0 diem,
tot hon sai -1).
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

from openai import AsyncOpenAI

SYSTEM_PROMPT = "Ban la tro ly phan tich du lieu ban hang. Chi tra ve mot khoi code Python."  # TODO prompt that cua Luong A
CODE_TIMEOUT = 60
CONCURRENCY_PER_GPU = 16


def load_questions(input_dir):
    # [ASSUMPTION] /data/questions.jsonl, moi dong: {"id","question","shop_id"...}
    path = os.path.join(input_dir, "questions.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_messages(q, input_dir):
    # TODO: nhet rulebook.md + README + schema ho so tiem (file/sheet/cot/dong/khoang ngay).
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q["question"]},
    ]


def extract_code(text):
    if "```" not in text:
        return text
    part = text.split("```", 2)[1]
    return part[6:] if part.startswith("python") else part


def run_code(code, input_dir):
    # Code phai in ra DUNG 1 dong la dap an cuoi. Chay trong process rieng, co timeout.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=CODE_TIMEOUT, cwd=input_dir)
        out = p.stdout.strip().splitlines()
        return (out[-1] if out and p.returncode == 0 else None), p.stderr[-500:]
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        os.unlink(path)


async def solve(q, client, sem, args, logf):
    async with sem:
        t0 = time.time()
        answer, err, raw = None, "", ""
        try:
            r = await client.chat.completions.create(
                model="qwen3.5-9b",
                messages=build_messages(q, args.input_dir),
                temperature=0.0,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = r.choices[0].message.content or ""
            answer, err = await asyncio.to_thread(run_code, extract_code(raw), args.input_dir)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        rec = {"id": q.get("id"), "answer": answer if answer is not None else "",
               "ok": answer is not None, "sec": round(time.time() - t0, 2), "err": err}
        logf.write(json.dumps({**rec, "question": q.get("question"), "raw": raw},
                              ensure_ascii=False) + "\n")
        logf.flush()
        return rec


async def main(args):
    questions = load_questions(args.input_dir)
    clients = [AsyncOpenAI(base_url=ep, api_key="EMPTY", timeout=600) for ep in args.endpoints]
    sem = asyncio.Semaphore(CONCURRENCY_PER_GPU * len(clients))
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.log_file + ".questions.jsonl", "a", encoding="utf-8") as logf:
        results = await asyncio.gather(*[
            solve(q, clients[i % len(clients)], sem, args, logf)
            for i, q in enumerate(questions)])
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            # [ASSUMPTION] khoa "id" va "answer"; sua theo dac ta submission cua BTC.
            f.write(json.dumps({"id": r["id"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
    print(f"Done: {sum(r['ok'] for r in results)}/{len(results)} answered")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--log_file", required=True)
    ap.add_argument("--endpoints", nargs="+", required=True)
    asyncio.run(main(ap.parse_args()))
