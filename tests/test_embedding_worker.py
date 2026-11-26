from __future__ import annotations

import socket
from types import SimpleNamespace

from megalm.gateway.protocol import FrameType, receive_frame
from megalm.gateway.worker import ReplicaWorker, WorkerMetrics


def _worker() -> tuple[ReplicaWorker, socket.socket]:
    sender, receiver = socket.socketpair()
    worker = ReplicaWorker.__new__(ReplicaWorker)
    worker.channel = sender
    worker.metrics = WorkerMetrics()
    worker.embedding_engine = SimpleNamespace(
        encode=lambda ids, **kwargs: [[float(len(row)), 0.5] for row in ids]
    )
    return worker, receiver


def test_embedding_frame_returns_vectors_in_input_order() -> None:
    worker, receiver = _worker()
    try:
        worker._handle_embedding(
            {
                "command_id": 42,
                "submissions": [
                    {"request_id": "a", "token_ids": [1, 2]},
                    {"request_id": "b", "token_ids": [3]},
                ],
                "dimensions": 2,
            }
        )
        frame = receive_frame(receiver)
    finally:
        worker.channel.close()
        receiver.close()
    assert frame is not None
    kind, payload = frame
    assert kind is FrameType.EMBEDDING_RESULT
    assert payload["command_id"] == 42
    assert [item["request_id"] for item in payload["results"]] == ["a", "b"]
    assert [item["embedding"] for item in payload["results"]] == [[2.0, 0.5], [1.0, 0.5]]
    assert all(item["ok"] for item in payload["results"])


def test_embedding_frame_maps_failures_to_result_items() -> None:
    worker, receiver = _worker()

    def fail(ids, **kwargs):
        raise ValueError("bad input")

    worker.embedding_engine = SimpleNamespace(encode=fail)
    try:
        worker._handle_embedding(
            {
                "command_id": 7,
                "submissions": [{"request_id": "x", "token_ids": [1]}],
            }
        )
        frame = receive_frame(receiver)
    finally:
        worker.channel.close()
        receiver.close()
    assert frame is not None
    _, payload = frame
    assert payload["results"][0]["ok"] is False
    assert "bad input" in str(payload["results"][0]["error"])
