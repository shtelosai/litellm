"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
)

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.adapters import (
    twork_reasoning_roundtrip as reasoning_roundtrip,
)


def _process_all(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    for event in events:
        wrapper._process_event(event)
    return list(wrapper._chunk_queue)


class TestProcessEventTextDeltaWithoutOutputItemAdded:
    """Streams that skip response.output_item.added (e.g. LMStudio) must still
    open a text block before any delta and never emit index -1."""

    def test_process_event_synthesizes_content_block_start_before_delta(self):
        chunks = _process_all(
            [
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "Hel"},
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "lo"},
            ]
        )
        assert [c["type"] for c in chunks] == [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
        ]
        assert chunks[0]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks] == [0, 0, 0]
        assert chunks[1]["delta"] == {"type": "text_delta", "text": "Hel"}

    def test_process_event_delta_without_item_id_never_yields_negative_index(self):
        chunks = _process_all([{"type": "response.output_text.delta", "delta": "Hi"}])
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]


    def test_process_event_unregistered_item_id_opens_new_text_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning", "id": "rs_1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert chunks[1]["type"] == "content_block_start"
        assert chunks[1]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks[1:]] == [1, 1]

    def test_process_event_registered_item_id_does_not_synthesize_start(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "m1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]


def test_reasoning_output_item_done_adds_roundtrip_block():
    chunks = _process_all(
        [
            {
                "type": "response.output_item.added",
                "item": {"type": "reasoning", "id": "rs_stream_native"},
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_stream_native",
                    "encrypted_content": "encrypted-stream-native",
                    "summary": [],
                },
            },
        ]
    )
    redacted = [
        chunk
        for chunk in chunks
        if chunk["type"] == "content_block_start"
        and chunk["content_block"]["type"] == "redacted_thinking"
    ]
    assert len(redacted) == 1
    unpacked = reasoning_roundtrip.unpack_signature(
        redacted[0]["content_block"]["data"]
    )
    assert unpacked and unpacked[0]["encrypted_content"] == "encrypted-stream-native"


@pytest.mark.parametrize(
    ("event", "expected_error"),
    [
        (
            {
                "type": "error",
                "error": {
                    "code": "invalid_request_error",
                    "message": "Invalid request",
                },
            },
            "Responses API stream error (invalid_request_error): Invalid request",
        ),
        (
            {
                "type": "response.failed",
                "response": {
                    "error": {"code": "server_error", "message": "Upstream failed"}
                },
            },
            "Responses API stream error (server_error): Upstream failed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_terminal_error_event_is_propagated(event, expected_error):
    async def _stream():
        yield event

    wrapper = AnthropicResponsesStreamWrapper(responses_stream=_stream(), model="m")
    wrapper._sent_message_start = True

    with pytest.raises(ValueError, match=expected_error.replace("(", r"\(").replace(")", r"\)")):
        await wrapper.__anext__()
