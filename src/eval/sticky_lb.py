#!/usr/bin/env python3
"""Sticky load balancer in front of several identical vLLM replicas (scripts/tb2_cloud.sh, 2026-10-15).

Each chat request goes to backends[hash(first two messages) % n], so every turn of one agent trial lands on the same
replica and keeps its prefix cache; requests are forwarded byte for byte (streaming or not), nothing is rewritten.
The backend list is re-read from --backends-file (one host:port per line) on every request, so replicas can be added
while trials run. It sits between llm_proxy.py (which logs every request) and vLLM: point the endpoint file that
llm_proxy re-reads at this balancer.

    python sticky_lb.py --listen 127.0.0.1:8150 --backends-file ~/apex/endpoints/merged_v4_cloud.backends
"""
import argparse
import hashlib
import json
from pathlib import Path

import aiohttp
from aiohttp import web

HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


def backends(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def pick(body: bytes, pool: list[str]) -> str:
    try:
        msgs = json.loads(body).get("messages") or []
        key = json.dumps(msgs[:2], sort_keys=True)[:8000]
    except Exception:
        key = ""
    return pool[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(pool)]


async def handle(request: web.Request) -> web.StreamResponse:
    app = request.app
    pool = backends(app["file"])
    body = await request.read()
    target = pick(body, pool) if request.method == "POST" else pool[0]
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    async with app["session"].request(request.method, f"http://{target}{request.rel_url}", data=body or None,
                                      headers=headers) as up:
        resp = web.StreamResponse(status=up.status,
                                  headers={k: v for k, v in up.headers.items() if k.lower() not in HOP})
        await resp.prepare(request)
        async for chunk in up.content.iter_any():
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def start(app: web.Application) -> None:
    app["session"] = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                           timeout=aiohttp.ClientTimeout(total=None, sock_connect=30))


async def stop(app: web.Application) -> None:
    await app["session"].close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", required=True)
    ap.add_argument("--backends-file", required=True)
    a = ap.parse_args()
    app = web.Application(client_max_size=1 << 30, handler_args={"max_field_size": 1 << 16})
    app["file"] = Path(a.backends_file).expanduser()
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_route("*", "/{tail:.*}", handle)
    host, port = a.listen.rsplit(":", 1)
    web.run_app(app, host=host, port=int(port), access_log=None, handler_cancellation=True)


if __name__ == "__main__":
    main()
