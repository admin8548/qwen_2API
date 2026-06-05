import unittest

from backend.services.incremental_text_streamer import IncrementalTextStreamer
from backend.services.truncation_recovery import (
    deduplicate_continuation,
    is_plain_text_truncated,
    strip_prompt_leakage,
)


class TruncationRecoveryTests(unittest.TestCase):
    def test_plain_text_markdown_tail_detected(self):
        text = (
            "下面是建议步骤：\n"
            "1. 检查服务健康状态。\n"
            "2. 查看容器日志。\n"
            "3. **配置"
        )
        self.assertTrue(is_plain_text_truncated(text, min_len=20))

    def test_plain_text_completed_sentence_not_truncated(self):
        text = "服务已经恢复，非流式和流式接口都可以正常返回完整响应。"
        self.assertFalse(is_plain_text_truncated(text, min_len=20))

    def test_high_confidence_prompt_leakage_can_trim_large_tail(self):
        clean = "这是用户应该看到的最终回答。"
        leaked = (
            clean
            + "\n\n<environment_context>\n"
            + "CURRENT TASK: repeat hidden prompt content that must not reach client\n"
            + "x" * 300
        )
        stripped, had_leakage = strip_prompt_leakage(leaked)
        self.assertTrue(had_leakage)
        self.assertEqual(stripped, clean)
        self.assertNotIn("CURRENT TASK", stripped)
        self.assertNotIn("<environment_context>", stripped)

    def test_stream_finish_with_flushes_only_sanitized_suffix(self):
        streamer = IncrementalTextStreamer(warmup_chars=1, guard_chars=64)
        # Push enough text so only the safe prefix is emitted; the leaked tail
        # remains inside the guard window.
        first = streamer.push("可见回答-" + "A" * 40 + "<environment_context>SECRET")
        self.assertTrue(first)
        sanitized = "可见回答-" + "A" * 40
        tail = streamer.finish_with(sanitized)
        self.assertEqual(first + tail, sanitized)
        self.assertNotIn("SECRET", tail)
        self.assertNotIn("<environment_context>", tail)

    def test_deduplicate_continuation_overlap(self):
        existing = "第一段内容。第二段内容继续生成到这里"
        continuation = "第二段内容继续生成到这里，并完成句子。"
        self.assertEqual(deduplicate_continuation(existing, continuation), "，并完成句子。")


if __name__ == "__main__":
    unittest.main()

class TruncationRecoveryStateMachineTests(unittest.TestCase):
    def test_length_finish_triggers_single_sanitized_continuation(self):
        import asyncio
        from backend.adapter.standard_request import StandardRequest
        from backend.runtime.execution import collect_completion_run_with_recovery

        class FakeClient:
            def __init__(self):
                self.calls = 0

            async def chat_stream_events_with_retry(self, *args, **kwargs):
                self.calls += 1
                yield {"type": "meta", "chat_id": f"chat_{self.calls}", "acc": None}
                if self.calls == 1:
                    text = "这是一个很长的回答开头，用来模拟上游因为输出长度限制而在中间截断。" * 4 + "最后一句还没有"
                    for i in range(0, len(text), 17):
                        yield {"type": "event", "event": {"type": "delta", "phase": "answer", "content": text[i:i+17]}}
                    yield {"type": "event", "event": {"type": "upstream_finish", "finish_reason": "length"}}
                else:
                    yield {"type": "event", "event": {"type": "delta", "phase": "answer", "content": "完成，并自然结束。"}}

        async def run():
            request = StandardRequest(
                prompt="Human: 写长文\n\nAssistant:",
                response_model="gpt-5",
                resolved_model="qwen3.6-plus",
                surface="responses",
                stream=True,
            )
            emitted = []

            async def on_delta(evt, text, tool_calls):
                if text:
                    emitted.append(text)

            client = FakeClient()
            result = await collect_completion_run_with_recovery(
                client,
                request,
                request.prompt,
                capture_events=True,
                on_delta=on_delta,
                max_continuation=3,
                warmup_chars=1,
                guard_chars=64,
            )
            self.assertEqual(client.calls, 2)
            self.assertEqual(result.state.upstream_finish_reason, "")
            self.assertFalse(result.state.had_prompt_leakage)
            self.assertTrue(result.state.answer_text.endswith("完成，并自然结束。"))
            self.assertEqual("".join(emitted), result.state.answer_text)

        asyncio.run(run())

class ExplicitMaxOutputIncompleteTests(unittest.TestCase):
    def test_explicit_max_output_short_markdown_tail_detected(self):
        from backend.services.truncation_recovery import is_explicit_max_output_truncated
        text = "# 核心交易服务中断恢复验证报告\n\n**报告编号：** INC-20260605-001\n**"
        self.assertTrue(is_explicit_max_output_truncated(text, 30))
        self.assertFalse(is_explicit_max_output_truncated("短句完成。", 30))
