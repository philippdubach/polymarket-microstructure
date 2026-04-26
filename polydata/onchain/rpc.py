"""Minimal JSON-RPC client for Polygon: eth_getLogs + finalized-block resolver."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class PolygonRpcClient:
    def __init__(
        self,
        url: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PolygonRpcClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call(self, method: str, params: list) -> Any:
        r = self._client.post(self.url, json={
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        })
        if r.status_code >= 400:
            raise RuntimeError(
                f"{method} HTTP {r.status_code}: {r.text[:500]}"
            )
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"{method} rpc error: {data['error']}")
        return data["result"]

    def block_number(self) -> int:
        return int(self._call("eth_blockNumber", []), 16)

    def finalized_block_number(self) -> int:
        """Polygon Heimdall-finalized block. Requires provider that honors
        the `finalized` tag (public polygon-rpc.com does; some free tiers
        do not)."""
        result = self._call("eth_getBlockByNumber", ["finalized", False])
        return int(result["number"], 16)

    def block_timestamp(self, block_number: int) -> int:
        result = self._call("eth_getBlockByNumber", [hex(block_number), False])
        return int(result["timestamp"], 16)

    def get_logs(
        self,
        address: str,
        topics: list[str | None],
        from_block: int,
        to_block: int,
        chunk: int = 1000,
    ) -> Iterator[dict]:
        start = from_block
        while start <= to_block:
            end = min(start + chunk - 1, to_block)
            params = [{
                "address": address, "topics": topics,
                "fromBlock": hex(start), "toBlock": hex(end),
            }]
            batch = self._call("eth_getLogs", params)
            yield from batch
            start = end + 1
