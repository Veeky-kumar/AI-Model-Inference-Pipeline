#!/usr/bin/env python3
"""
Load test script — trigger HPA by hammering the inference endpoint.
Usage:
  pip install httpx
  python scripts/load_test.py --url http://localhost:8080 --rps 200 --duration 60
"""

import asyncio
import httpx
import argparse
import time
import statistics
from datetime import datetime


PAYLOAD = {
    "inputs": [{
        "name": "input",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [5.1, 3.5, 1.4, 0.2]
    }]
}

async def send_request(client: httpx.AsyncClient, url: str, results: list):
    start = time.time()
    try:
        resp = await client.post(f"{url}/v2/models/iris-classifier/infer", json=PAYLOAD, timeout=5.0)
        duration = time.time() - start
        results.append({"status": resp.status_code, "duration": duration, "ok": resp.status_code == 200})
    except Exception as e:
        results.append({"status": 0, "duration": time.time() - start, "ok": False, "error": str(e)})

async def run_load_test(url: str, rps: int, duration: int):
    results = []
    interval = 1.0 / rps
    end_time = time.time() + duration

    print(f"\n🚀 Load test started")
    print(f"   Target: {url}")
    print(f"   Rate: {rps} req/s for {duration}s\n")

    async with httpx.AsyncClient() as client:
        tasks = []
        while time.time() < end_time:
            task = asyncio.create_task(send_request(client, url, results))
            tasks.append(task)
            await asyncio.sleep(interval)

        await asyncio.gather(*tasks)

    # ── Report ────────────────────────────────────────────────────────────────
    total = len(results)
    success = sum(1 for r in results if r["ok"])
    latencies = [r["duration"] for r in results if r["ok"]]

    print("=" * 50)
    print(f"  Total requests:  {total}")
    print(f"  Successes:       {success} ({100*success/total:.1f}%)")
    print(f"  Failures:        {total - success}")
    if latencies:
        print(f"  Latency p50:     {statistics.median(latencies)*1000:.1f}ms")
        print(f"  Latency p95:     {sorted(latencies)[int(0.95*len(latencies))]*1000:.1f}ms")
        print(f"  Latency p99:     {sorted(latencies)[int(0.99*len(latencies))]*1000:.1f}ms")
        print(f"  Max latency:     {max(latencies)*1000:.1f}ms")
    print("=" * 50)
    print(f"\n📊 Watch HPA scale up:")
    print(f"   kubectl get hpa -n ai-inference -w")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the inference server")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--rps", type=int, default=50, help="Requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    args = parser.parse_args()

    asyncio.run(run_load_test(args.url, args.rps, args.duration))
