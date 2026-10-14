#!/usr/bin/env python3
"""OpenAI-compatible relay between eval harnesses and the vLLM serve job.

Why: (1) the GPU serve job may be preempted/requeued onto another node; the relay
re-reads the endpoint file on every request and, on connection failure, waits for
the endpoint to come back (up to --wait-sec) instead of failing the trial;
(2) it logs, for every request, the sampling parameters the harness actually sent
plus token usage, so "identical harness config across models" is verifiable and
cost is measurable per run. The relay never modifies requests or responses.

Listens on TCP and optionally on a Unix socket (the ASI-Bench net-locked sandbox
reaches it only through that socket). Requires aiohttp (present in envs/eval).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp
from aiohttp import web

LOG_KEYS = ("model", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
            "frequency_penalty", "repetition_penalty", "max_tokens", "max_completion_tokens",
            "chat_template_kwargs", "reasoning_effort", "stream", "seed", "n", "tool_choice")
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


class Endpoint:
    def __init__(self, path: Path):
        self.path, self._mtime, self.base = path, None, None

    def get(self) -> str | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        if st.st_mtime != self._mtime:
            kv = {}
            for line in self.path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip().strip("'\"")
            self.base = f"http://{kv['ENDPOINT_HOST']}:{kv['ENDPOINT_PORT']}"
            self._mtime = st.st_mtime
        return self.base


def summarize_request(body: bytes) -> dict:
    try:
        j = json.loads(body)
    except Exception:
        return {}
    s = {k: j[k] for k in LOG_KEYS if k in j}
    eb = j.get("extra_body") or {}
    s.update({f"extra_body.{k}": v for k, v in eb.items() if k in LOG_KEYS})
    s["n_messages"] = len(j.get("messages") or [])
    s["n_tools"] = len(j.get("tools") or [])
    return s


async def main_async(a: argparse.Namespace) -> None:
    ep = Endpoint(Path(a.endpoint_file))
    logf = open(a.log, "a", buffering=1) if a.log else None
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=a.read_timeout)
    session = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)

    def log(rec: dict, t0: float) -> None:
        rec["latency_s"] = round(time.time() - t0, 3)
        if logf:
            logf.write(json.dumps(rec) + "\n")

    async def forward(req: web.Request, base: str, body: bytes, headers: dict, rec: dict):
        async with session.request(req.method, base + req.path_qs, data=body, headers=headers) as up:
            rec.update(upstream=base, status=up.status)
            out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP}
            if "text/event-stream" in up.headers.get("content-type", ""):
                resp = web.StreamResponse(status=up.status, headers=out_headers)
                await resp.prepare(req)
                nbytes, tail = 0, b""
                async for chunk in up.content.iter_any():
                    nbytes += len(chunk)
                    tail = (tail + chunk)[-8192:]
                    await resp.write(chunk)
                await resp.write_eof()
                rec["stream_bytes"] = nbytes
                for line in reversed(tail.split(b"\n")):  # final usage chunk, if requested
                    if line.startswith(b"data:") and b'"usage"' in line:
                        try:
                            rec["usage"] = json.loads(line[5:]).get("usage")
                        except Exception:
                            pass
                        break
                return resp
            data = await up.read()
            if "json" in up.headers.get("content-type", ""):
                try:
                    j = json.loads(data)
                    rec["usage"] = j.get("usage")
                    rec["finish_reason"] = ((j.get("choices") or [{}])[0]).get("finish_reason")
                except Exception:
                    pass
            return web.Response(status=up.status, body=data, headers=out_headers)

    async def handle(req: web.Request) -> web.StreamResponse:
        body = await req.read()
        t0, waited = time.time(), 0
        rec = {"ts": t0, "path": req.path, **summarize_request(body)}
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
        while True:
            base = ep.get()
            if base is not None:
                try:
                    resp = await forward(req, base, body, headers, rec)
                    rec["waited_s"] = waited
                    log(rec, t0)
                    return resp
                except aiohttp.ClientConnectorError as e:  # nothing sent upstream yet: safe to retry
                    rec["last_error"] = f"{type(e).__name__}: {e}"
                except Exception as e:  # mid-request failure: report, never replay
                    rec["error"] = f"{type(e).__name__}: {e}"
                    log(rec, t0)
                    if rec.get("status") is not None:
                        raise
                    return web.json_response({"error": rec["error"]}, status=502)
            if waited >= a.wait_sec:
                rec["error"] = f"endpoint unavailable after {waited}s: {rec.get('last_error')}"
                log(rec, t0)
                return web.json_response({"error": rec["error"]}, status=502)
            await asyncio.sleep(10)
            waited += 10

    app = web.Application(client_max_size=1024 ** 3)
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    host, port = a.listen.rsplit(":", 1)
    await web.TCPSite(runner, host, int(port)).start()
    if a.uds:
        Path(a.uds).unlink(missing_ok=True)
        await web.UnixSite(runner, a.uds).start()
        os.chmod(a.uds, 0o600)
    print(f"llm_proxy listening on {a.listen} uds={a.uds}", flush=True)
    await asyncio.Event().wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-file", required=True)
    ap.add_argument("--listen", default="127.0.0.1:18000")
    ap.add_argument("--uds", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--wait-sec", type=float, default=1800.0,
                    help="how long to wait for a (re)starting endpoint before 502")
    ap.add_argument("--read-timeout", type=float, default=7200.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
