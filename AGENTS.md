Read @CLAUDE.md for coding guidelines

Anthropic `/v1/messages` 转 OpenAI Responses API 时，两条适配路径都必须在 `LITELLM_TWORK_REASONING_ROUNDTRIP` 启用时合并 `include: ["reasoning.encrypted_content"]`，并保留现有 `include` 项，确保加密推理可以跨轮往返。
Anthropic 流式合成 `redacted_thinking` 后，任何 `content_block_delta` 都只能发给仍处于打开状态的同索引块；同步和异步回归测试必须用块状态机校验 start / delta / stop 顺序。
