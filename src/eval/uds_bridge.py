#!/usr/bin/env python3
"""TCP -> Unix-socket byte relay (stdlib only).

Used inside the ASI-Bench network-locked namespace (`unshare --net`: loopback only):
agents connect to 127.0.0.1:<port>, bytes are relayed to the llm_proxy's Unix socket,
which lives on the host side. This is the ONLY path out of the namespace, so the agent
can reach the model endpoint and nothing else (no PyPI, no HuggingFace, no web).
"""
import argparse
import asyncio


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


async def main(listen: str, uds: str) -> None:
    async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        try:
            ur, uw = await asyncio.open_unix_connection(uds)
        except OSError:
            cw.close()
            return
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))

    host, port = listen.rsplit(":", 1)
    server = await asyncio.start_server(handle, host, int(port))
    print(f"uds_bridge {listen} -> {uds}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:18000")
    ap.add_argument("--uds", required=True)
    a = ap.parse_args()
    asyncio.run(main(a.listen, a.uds))
