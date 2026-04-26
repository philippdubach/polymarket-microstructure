import httpx

from polydata.onchain.rpc import PolygonRpcClient


class MockTransport(httpx.BaseTransport):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def handle_request(self, request):
        self.calls.append(request)
        response = self.responses.pop(0)
        return httpx.Response(200, json=response)


def test_block_number_latest():
    transport = MockTransport([{"jsonrpc": "2.0", "id": 1, "result": "0x1ABCD"}])
    client = PolygonRpcClient("https://mock/rpc", transport=transport)
    assert client.block_number() == 0x1ABCD


def test_block_number_finalized():
    transport = MockTransport([{"jsonrpc": "2.0", "id": 1,
                                "result": {"number": "0x1AAAA", "timestamp": "0x6641"}}])
    client = PolygonRpcClient("https://mock/rpc", transport=transport)
    assert client.finalized_block_number() == 0x1AAAA


def test_get_logs_paginates_by_chunk_including_empty_middle():
    # 0-999 (1 log), 1000-1999 (empty), 2000-2000 (1 log).
    # get_logs must NOT break on empty; must cover the full range.
    transport = MockTransport([
        {"jsonrpc": "2.0", "id": 1, "result": [{
            "blockNumber": "0x10", "topics": ["0xabc"],
            "data": "0x00", "logIndex": "0x0", "transactionHash": "0xaa"}]},
        {"jsonrpc": "2.0", "id": 1, "result": []},
        {"jsonrpc": "2.0", "id": 1, "result": [{
            "blockNumber": "0x7d0", "topics": ["0xabc"],
            "data": "0x00", "logIndex": "0x1", "transactionHash": "0xbb"}]},
    ])
    client = PolygonRpcClient("https://mock/rpc", transport=transport)
    logs = list(client.get_logs(
        address="0xExch", topics=["0xabc"],
        from_block=0, to_block=2000, chunk=1000,
    ))
    assert len(logs) == 2
    assert len(transport.calls) == 3  # must call all 3 chunks
