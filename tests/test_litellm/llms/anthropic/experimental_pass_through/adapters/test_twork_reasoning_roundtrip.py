"""Twork reasoning roundtrip tests (upstream issue #24425).

Covers:
- codec pack/unpack roundtrip, malformed-signature tolerance, empty-ec skip
- env kill-switch bypass on all three legs
- request leg: marked thinking blocks -> assistant reasoning_items (and NOT
  forwarded as thinking_blocks); real thinking blocks unchanged; multi-item
- response legs: non-streaming content synthesis; streaming wrapper synthesis
  (sync + async), at-most-once, SSE ordering before message_delta
- leak-by-construction: streaming synthesis lives in AnthropicStreamWrapper
  (anthropic-only); interleaved marked/unmarked streams stay isolated
"""

import asyncio
import json

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters import (
    twork_reasoning_roundtrip as rt,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)

REASONING_ITEMS = [
    {
        "id": "rs_test001",
        "type": "reasoning",
        "encrypted_content": "gAAAAA-test-encrypted-payload-001",
        "summary": [{"type": "summary_text", "text": "plan the tool call"}],
    },
    {
        "id": "rs_test002",
        "type": "reasoning",
        "encrypted_content": "gAAAAA-test-encrypted-payload-002",
        "summary": [],
    },
]


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------


def test_codec_roundtrip_multi_item():
    sig = rt.pack_reasoning_items(REASONING_ITEMS)
    assert sig is not None and sig.startswith(rt.SIGNATURE_PREFIX)
    items = rt.unpack_signature(sig)
    assert items is not None and len(items) == 2
    assert items[0]["id"] == "rs_test001"
    assert items[0]["encrypted_content"] == "gAAAAA-test-encrypted-payload-001"
    assert items[1]["encrypted_content"] == "gAAAAA-test-encrypted-payload-002"
    assert all(i["type"] == "reasoning" for i in items)


def test_codec_skips_items_without_encrypted_content():
    assert rt.pack_reasoning_items([{"id": "rs_x", "encrypted_content": None}]) is None
    assert rt.pack_reasoning_items([]) is None
    assert rt.pack_reasoning_items(None) is None


def test_codec_tolerates_malformed_signatures():
    assert rt.unpack_signature(None) is None
    assert rt.unpack_signature("") is None
    assert rt.unpack_signature("not-marked") is None
    assert rt.unpack_signature(rt.SIGNATURE_PREFIX + "!!!not-base64!!!") is None
    # 合法 base64 但非 JSON 列表
    import base64

    assert rt.unpack_signature(rt.SIGNATURE_PREFIX + base64.b64encode(b'{"a":1}').decode()) is None


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("LITELLM_TWORK_REASONING_ROUNDTRIP", "0")
    assert rt.is_enabled() is False
    monkeypatch.delenv("LITELLM_TWORK_REASONING_ROUNDTRIP", raising=False)
    assert rt.is_enabled() is True


# ---------------------------------------------------------------------------
# request leg (anthropic messages -> openai chat messages)
# ---------------------------------------------------------------------------


def _anthropic_history_with_marked_block():
    sig = rt.pack_reasoning_items(REASONING_ITEMS)
    return [
        {"role": "user", "content": "查上海天气"},
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": sig},
                {"type": "text", "text": "我来查一下。"},
                {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "上海"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "晴 31 度"}],
        },
    ]


def test_request_leg_restores_reasoning_items():
    adapter = LiteLLMAnthropicMessagesAdapter()
    out = adapter.translate_anthropic_messages_to_openai(messages=_anthropic_history_with_marked_block())
    assistant = next(m for m in out if m["role"] == "assistant")
    items = assistant.get("reasoning_items")
    assert items and len(items) == 2
    assert items[0]["encrypted_content"] == "gAAAAA-test-encrypted-payload-001"
    # 合成块不得作为真 thinking/redacted 块转发
    for tb in assistant.get("thinking_blocks") or []:
        assert not str(tb.get("data", "")).startswith(rt.SIGNATURE_PREFIX)
        assert not str(tb.get("signature", "")).startswith(rt.SIGNATURE_PREFIX)


def test_request_leg_keeps_real_thinking_blocks():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "real thought", "signature": "sig-from-claude"},
                {"type": "text", "text": "ok"},
            ],
        },
        {"role": "user", "content": "again"},
    ]
    out = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(messages=messages)
    assistant = next(m for m in out if m["role"] == "assistant")
    tbs = assistant.get("thinking_blocks")
    assert tbs and tbs[0]["signature"] == "sig-from-claude"
    assert not assistant.get("reasoning_items")


def test_request_leg_disabled_leaves_marked_block_as_thinking(monkeypatch):
    monkeypatch.setenv("LITELLM_TWORK_REASONING_ROUNDTRIP", "0")
    out = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=_anthropic_history_with_marked_block()
    )
    assistant = next(m for m in out if m["role"] == "assistant")
    # 关闭后走原版路径：不产生 reasoning_items，标记块按普通 redacted_thinking 处理（不崩溃、不 400）
    assert not assistant.get("reasoning_items")
    tbs = assistant.get("thinking_blocks") or []
    assert any(str(tb.get("data", "")).startswith(rt.SIGNATURE_PREFIX) for tb in tbs)


# ---------------------------------------------------------------------------
# response leg: non-streaming
# ---------------------------------------------------------------------------


def _chat_choice_with_reasoning_items():
    msg = Message(role="assistant", content="上海今天晴。", reasoning_items=REASONING_ITEMS)
    return [Choices(index=0, message=msg, finish_reason="stop")]


def test_nonstream_response_leg_synthesizes_marked_block():
    adapter = LiteLLMAnthropicMessagesAdapter()
    content = adapter._translate_openai_content_to_anthropic(choices=_chat_choice_with_reasoning_items())
    redacted = [b for b in content if b.get("type") == "redacted_thinking"]
    assert len(redacted) == 1
    items = rt.unpack_signature(redacted[0]["data"])
    assert items and len(items) == 2
    # 文本块保持
    assert any(b.get("type") == "text" and b.get("text") == "上海今天晴。" for b in content)


def test_nonstream_response_leg_disabled(monkeypatch):
    monkeypatch.setenv("LITELLM_TWORK_REASONING_ROUNDTRIP", "0")
    content = LiteLLMAnthropicMessagesAdapter()._translate_openai_content_to_anthropic(
        choices=_chat_choice_with_reasoning_items()
    )
    assert not any(
        str(b.get("data", "")).startswith(rt.SIGNATURE_PREFIX) for b in content if b.get("type") == "redacted_thinking"
    )


# ---------------------------------------------------------------------------
# response leg: streaming wrapper (anthropic-only by construction)
# ---------------------------------------------------------------------------


class _ListStream:
    """最下游 transport 的最小替身：仅提供 chunk 迭代，不 mock 任何转换器。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __iter__(self):
        return iter(list(self._chunks))

    def __aiter__(self):
        self._ait = iter(list(self._chunks))
        return self

    async def __anext__(self):
        try:
            return next(self._ait)
        except StopIteration:
            raise StopAsyncIteration


def _bridge_like_chunks(with_reasoning=True, text="上海今天晴。"):
    """构造 responses 桥输出形态的 chat 流：文本增量 + 单一终块（finish+usage+reasoning_items）。"""
    text_chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=text), finish_reason=None)]
    )
    final = ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content="", reasoning_items=(REASONING_ITEMS if with_reasoning else None)),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    return [text_chunk, final]


def _customstreamwrapper_shape_chunks(text="上海今天晴。"):
    """CustomStreamWrapper 重建后的真实形态：内容块 → reasoning-only 块 → finish 块 → 空尾块。"""
    text_chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=text), finish_reason=None)]
    )
    reasoning_only = ModelResponseStream(
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content="", reasoning_items=REASONING_ITEMS),
                finish_reason=None,
            )
        ]
    )
    finish_chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=""), finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    trailing_empty = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=""), finish_reason=None)]
    )
    return [text_chunk, reasoning_only, finish_chunk, trailing_empty]


def _collect_events_sync(chunks):
    wrapper = AnthropicStreamWrapper(completion_stream=_ListStream(chunks), model="gpt-5.6-sol")
    events = []
    for raw in wrapper.anthropic_sse_wrapper():
        s = raw.decode() if isinstance(raw, bytes) else raw
        for line in s.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


def _collect_events_async(chunks):
    async def _run():
        wrapper = AnthropicStreamWrapper(completion_stream=_ListStream(chunks), model="gpt-5.6-sol")
        events = []
        async for raw in wrapper.async_anthropic_sse_wrapper():
            s = raw.decode() if isinstance(raw, bytes) else raw
            for line in s.splitlines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
        return events

    return asyncio.run(_run())


def _assert_marked_thinking_stream(events):
    starts = [
        e
        for e in events
        if e["type"] == "content_block_start" and e["content_block"]["type"] == "redacted_thinking"
    ]
    assert len(starts) == 1, f"应恰好一个合成 redacted_thinking 块: {events}"
    idx = starts[0]["index"]
    # redacted_thinking 的 data 直接挂在 content_block_start（无 delta）
    items = rt.unpack_signature(starts[0]["content_block"]["data"])
    assert items and len(items) == 2 and items[0]["encrypted_content"] == "gAAAAA-test-encrypted-payload-001"
    # 该块必须闭合，且先于 message_delta
    assert any(e["type"] == "content_block_stop" and e["index"] == idx for e in events)
    md_pos = next(i for i, e in enumerate(events) if e["type"] == "message_delta")
    start_pos = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "content_block_start" and e["content_block"]["type"] == "redacted_thinking"
    )
    assert start_pos < md_pos, "合成 redacted_thinking 块必须先于 message_delta"

    open_blocks = set()
    for event in events:
        event_type = event["type"]
        if event_type == "content_block_start":
            assert event["index"] not in open_blocks, f"content block 重复开始: {events}"
            open_blocks.add(event["index"])
        elif event_type == "content_block_delta":
            assert event["index"] in open_blocks, f"已关闭或未开始的 content block 收到 delta: {events}"
        elif event_type == "content_block_stop":
            assert event["index"] in open_blocks, f"content block 未开始便停止或重复停止: {events}"
            open_blocks.remove(event["index"])
    assert not open_blocks, f"流结束后仍有未关闭的 content block: {events}"


def test_streaming_leg_sync_synthesizes_marked_block():
    _assert_marked_thinking_stream(_collect_events_sync(_bridge_like_chunks()))


def test_streaming_leg_async_synthesizes_marked_block():
    _assert_marked_thinking_stream(_collect_events_async(_bridge_like_chunks()))


def test_streaming_leg_customstreamwrapper_shape_sync():
    """生产真实链路形态（finish 与 reasoning_items 被 CustomStreamWrapper 拆分）也必须合成。"""
    _assert_marked_thinking_stream(_collect_events_sync(_customstreamwrapper_shape_chunks()))


def test_streaming_leg_customstreamwrapper_shape_async():
    _assert_marked_thinking_stream(_collect_events_async(_customstreamwrapper_shape_chunks()))


def test_streaming_leg_no_reasoning_items_no_synthesis():
    events = _collect_events_sync(_bridge_like_chunks(with_reasoning=False))
    assert not any(
        e["type"] == "content_block_start" and e["content_block"]["type"] == "redacted_thinking" for e in events
    )


def test_streaming_leg_disabled(monkeypatch):
    monkeypatch.setenv("LITELLM_TWORK_REASONING_ROUNDTRIP", "0")
    events = _collect_events_sync(_bridge_like_chunks())
    assert not any(
        e["type"] == "content_block_start" and e["content_block"]["type"] == "redacted_thinking" for e in events
    )


# ---------------------------------------------------------------------------
# bridge 迭代器：output_item.done 形态的 reasoning 累积（上游两种下发形态之一）
# ---------------------------------------------------------------------------


def test_bridge_iterator_accumulates_output_item_reasoning():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )

    it = OpenAiResponsesToChatCompletionStreamIterator(None, sync_stream=True)
    # reasoning 只出现在 output_item.done 事件（completed 输出不含）
    it.chunk_parser(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "rs_stream01",
                "type": "reasoning",
                "content": [],
                "encrypted_content": "EC-STREAM-01",
                "summary": [{"type": "summary_text", "text": "streamed"}],
            },
        }
    )
    final = it.chunk_parser(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_x",
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        }
    )
    assert final.choices[0].finish_reason is not None
    items = getattr(final.choices[0].delta, "reasoning_items", None)
    assert items and items[0]["id"] == "rs_stream01"
    assert items[0]["encrypted_content"] == "EC-STREAM-01"


def test_bridge_iterator_merges_without_duplicating_completed_items():
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        OpenAiResponsesToChatCompletionStreamIterator,
    )

    it = OpenAiResponsesToChatCompletionStreamIterator(None, sync_stream=True)
    it.chunk_parser(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"id": "rs_dup", "type": "reasoning", "content": [], "encrypted_content": "EC-DUP", "summary": []},
        }
    )
    # completed 输出里也带同 id 的 reasoning（另一种上游形态）→ 不应重复
    final = it.chunk_parser(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_y",
                "status": "completed",
                "output": [
                    {"id": "rs_dup", "type": "reasoning", "content": [], "encrypted_content": "EC-DUP", "summary": []},
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        }
    )
    items = getattr(final.choices[0].delta, "reasoning_items", None)
    assert items is not None
    assert [r["id"] for r in items].count("rs_dup") == 1


# ---------------------------------------------------------------------------
# 跨层可达性 + 并发隔离（Codex 闸门测试，按本设计等价形式）：
# 设计上"标记通道"= AnthropicStreamWrapper 本身（仅 anthropic 路径构造它），
# 共享桥零改动。等价闸门：两条流并发交错（Event 强制重叠、覆盖两种终块顺序），
# anthropic wrapper 流产出合成块，而"无 wrapper 的普通 chat 流"逐字段等于原始 chunk。
# ---------------------------------------------------------------------------


class _GatedStream:
    """用 asyncio.Event 控制终块放行时序的 transport 替身。"""

    def __init__(self, chunks, gate: "asyncio.Event"):
        self._chunks = list(chunks)
        self._gate = gate

    def __aiter__(self):
        self._i = 0
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        is_final = chunk.choices[0].finish_reason is not None
        if is_final:
            await self._gate.wait()
        self._i += 1
        return chunk


@pytest.mark.parametrize("marked_final_first", [True, False])
def test_interleaved_marked_anthropic_and_plain_chat_streams(marked_final_first):
    async def _run():
        gate_marked = asyncio.Event()
        gate_plain = asyncio.Event()

        # anthropic 路径：真实 wrapper（除 transport 外全真实链路）
        marked_chunks = _bridge_like_chunks(with_reasoning=True)
        wrapper = AnthropicStreamWrapper(
            completion_stream=_GatedStream(marked_chunks, gate_marked), model="gpt-5.6-sol"
        )

        async def consume_marked():
            events = []
            async for raw in wrapper.async_anthropic_sse_wrapper():
                s = raw.decode() if isinstance(raw, bytes) else raw
                for line in s.splitlines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:].strip()))
            return events

        # 普通 chat 路径：无 anthropic wrapper —— 客户端直接消费 chat chunk
        plain_chunks = _bridge_like_chunks(with_reasoning=True)

        async def consume_plain():
            out = []
            async for chunk in _GatedStream(plain_chunks, gate_plain):
                out.append(chunk)
            return out

        t_marked = asyncio.create_task(consume_marked())
        t_plain = asyncio.create_task(consume_plain())
        # 强制重叠窗口：两条流都已消费完非终块、终块被闸住
        await asyncio.sleep(0.05)
        # 覆盖两种终块到达顺序
        if marked_final_first:
            gate_marked.set()
            await asyncio.sleep(0.02)
            gate_plain.set()
        else:
            gate_plain.set()
            await asyncio.sleep(0.02)
            gate_marked.set()

        marked_events = await t_marked
        plain_out = await t_plain
        return marked_events, plain_out, plain_chunks

    marked_events, plain_out, plain_chunks = asyncio.run(_run())

    # anthropic 流：产出恰一个合成块
    _assert_marked_thinking_stream(marked_events)

    # 普通 chat 流：chunk 逐字段等于原始（无任何 tworkrs 痕迹注入）
    assert len(plain_out) == len(plain_chunks)
    for got, want in zip(plain_out, plain_chunks):
        assert got is want  # transport 原样传递，未被任何层改写
        dumped = got.model_dump()
        assert rt.SIGNATURE_PREFIX not in json.dumps(dumped, default=str)
        assert not (dumped["choices"][0]["delta"].get("thinking_blocks"))
