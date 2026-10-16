"""vLLM 冒烟测试：文本、thinking、工具调用、HLE 图像题各发一次请求。
用法：python smoke_serve.py [--base-url http://127.0.0.1:8000/v1]"""
import argparse, json, time, pandas as pd
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--hle", default="/scratch/eh3323/apex/data/benchmarks/hle/data/test-00000-of-00001.parquet")
a = ap.parse_args()
c = OpenAI(base_url=a.base_url, api_key="x")
M = c.models.list().data[0].id
print("model:", M)

def run(name, **kw):
    t = time.time()
    r = c.chat.completions.create(model=M, max_tokens=kw.pop("max_tokens", 2048), **kw)
    m = r.choices[0].message
    rc = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
    print(f"\n=== {name}  ({time.time()-t:.1f}s, usage={r.usage.completion_tokens} tok)")
    print("reasoning:", (rc or "")[:300].replace("\n", " "))
    print("content  :", (m.content or "")[:300].replace("\n", " "))
    if m.tool_calls: print("tool_calls:", [(x.function.name, x.function.arguments) for x in m.tool_calls])

# 1) 文本，关闭 thinking
run("text/no-think", messages=[{"role": "user", "content": "What is 17*23? Answer with the number only."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
# 2) 文本，开启 thinking
run("text/think", messages=[{"role": "user", "content": "What is 17*23?"}], max_tokens=4096)
# 3) 工具调用
tools = [{"type": "function", "function": {"name": "bash", "description": "run a shell command",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
run("tool-call", messages=[{"role": "user", "content": "List files in /tmp using the bash tool."}], tools=tools,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
# 4) HLE 带图题（检验多模态）
d = pd.read_parquet(a.hle); row = d[d["image"].fillna("") != ""].iloc[0]
run("hle-image", messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one sentence."},
        {"type": "image_url", "image_url": {"url": row["image"]}}]}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print("\nSMOKE OK")
