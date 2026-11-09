import argparse, asyncio, json, os, subprocess, sys, tempfile, time
from openai import AsyncOpenAI

SYS = "Ban la tro ly phan tich du lieu ban hang. Chi tra ve mot khoi code Python."  # TODO prompt luong A
PER_GPU = 16


def load(d):
    # TODO: sua theo format that cua BTC
    with open(os.path.join(d, "questions.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def msgs(q, d):
    # TODO: them rulebook, README, schema tiem
    return [{"role": "system", "content": SYS}, {"role": "user", "content": q["question"]}]


def get_code(t):
    if "```" not in t:
        return t
    c = t.split("```", 2)[1]
    return c[6:] if c.startswith("python") else c


def run(code, d):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        p = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=60, cwd=d)
        out = p.stdout.strip().splitlines()
        return (out[-1] if out and p.returncode == 0 else None), p.stderr[-500:]
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        os.unlink(f.name)


async def solve(q, cl, sem, a, log):
    async with sem:
        t, ans, err, raw = time.time(), None, "", ""
        try:
            r = await cl.chat.completions.create(
                model="qwen3.5-9b", messages=msgs(q, a.input_dir), temperature=0.0, max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            raw = r.choices[0].message.content or ""
            ans, err = await asyncio.to_thread(run, get_code(raw), a.input_dir)
        except Exception as e:
            err = repr(e)
        log.write(json.dumps({"id": q.get("id"), "q": q.get("question"), "raw": raw, "ans": ans,
                              "err": err, "sec": round(time.time() - t, 2)}, ensure_ascii=False) + "\n")
        log.flush()
        return q.get("id"), ans


async def main(a):
    qs = load(a.input_dir)
    cls = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=600) for u in a.endpoints]
    sem = asyncio.Semaphore(PER_GPU * len(cls))
    os.makedirs(os.path.dirname(a.output_file), exist_ok=True)
    with open(a.log_file + ".questions.jsonl", "a", encoding="utf-8") as log:
        res = await asyncio.gather(*[solve(q, cls[i % len(cls)], sem, a, log) for i, q in enumerate(qs)])
    with open(a.output_file, "w", encoding="utf-8") as f:
        for i, ans in res:  # loi => de trong (0 diem, tot hon sai -1)
            f.write(json.dumps({"id": i, "answer": ans or ""}, ensure_ascii=False) + "\n")
    print(f"{sum(bool(a) for _, a in res)}/{len(res)} answered")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--log_file", required=True)
    p.add_argument("--endpoints", nargs="+", required=True)
    asyncio.run(main(p.parse_args()))
