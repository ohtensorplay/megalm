import socket
from types import SimpleNamespace

from megalm.contracts.request import FinishReason
from megalm.gateway.protocol import FrameType, output_from_wire, receive_frame
from megalm.gateway.worker import ReplicaWorker, WorkerMetrics
from megalm.runtime.scheduler import Completion


def test_worker_coalesces_speculative_tokens_into_one_stream_output() -> None:
    sender, receiver = socket.socketpair()
    forgotten: list[str] = []
    worker = ReplicaWorker.__new__(ReplicaWorker)
    worker.channel = sender
    worker.metrics = WorkerMetrics()
    worker._stream_ids = {"request": 7}
    worker._request_ids = {7: "request"}
    worker.engine = SimpleNamespace(
        scheduler=SimpleNamespace(forget=forgotten.append)
    )
    completion = Completion(
        events=(),
        finished_request_ids=("request",),
        output_request_ids=("request", "request", "request"),
        output_token_ids=(11, 12, 13),
        finish_reasons=(None, None, FinishReason.LENGTH),
        output_logprobs=(None, None, None),
    )
    try:
        worker._send_completion(completion)
        frame = receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()

    assert frame is not None
    frame_type, payload = frame
    assert frame_type is FrameType.OUTPUTS
    assert len(payload["outputs"]) == 1
    assert output_from_wire(payload["outputs"][0]) == (
        7,
        (11, 12, 13),
        (),
        FinishReason.LENGTH,
        None,
        (),
    )
    assert worker.metrics.emitted_events == 4
    assert forgotten == ["request"]


def test_worker_sends_label_logprobs_for_score_reads() -> None:
    sender, receiver = socket.socketpair()
    forgotten: list[str] = []
    worker = ReplicaWorker.__new__(ReplicaWorker)
    worker.channel = sender
    worker.metrics = WorkerMetrics()
    worker._stream_ids = {"score": 9}
    worker._request_ids = {9: "score"}
    worker.engine = SimpleNamespace(
        scheduler=SimpleNamespace(forget=forgotten.append)
    )
    completion = Completion(
        events=(),
        finished_request_ids=("score",),
        output_request_ids=("score",),
        output_token_ids=(7,),
        finish_reasons=(FinishReason.LENGTH,),
        output_logprobs=(None,),
        output_label_ids=((7, 9, 11),),
        output_label_logprobs=((-0.1, -2.0, -3.0),),
    )
    try:
        worker._send_completion(completion)
        frame = receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()

    assert frame is not None
    frame_type, payload = frame
    assert frame_type is FrameType.OUTPUTS
    assert output_from_wire(payload["outputs"][0]) == (
        9,
        (7, 9, 11),
        (),
        FinishReason.LENGTH,
        None,
        (-0.1, -2.0, -3.0),
    )
