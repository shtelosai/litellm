Read @CLAUDE.md for coding guidelines

Anthropic `/v1/messages` 转 OpenAI Responses API 时，两条适配路径都必须在 `LITELLM_TWORK_REASONING_ROUNDTRIP` 启用时合并 `include: ["reasoning.encrypted_content"]`，并保留现有 `include` 项，确保加密推理可以跨轮往返。
