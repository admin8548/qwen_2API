import json
import unittest

from backend.adapter.standard_request import StandardRequest
from backend.runtime.execution import RuntimeAttemptState, RuntimeExecutionResult
from backend.services.responses_adapter import adapt_responses_request_to_chat
from backend.services.responses_formatters import ResponsesStreamTranslator, build_responses_payload
from backend.services.standard_request_builder import build_chat_standard_request


class ResponsesAdapterTests(unittest.TestCase):
    def test_string_input_becomes_user_message(self):
        payload = adapt_responses_request_to_chat({"model": "gpt-4o-mini", "input": "hello"})
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])

    def test_message_function_call_and_output_items_convert_to_chat_history(self):
        payload = adapt_responses_request_to_chat(
            {
                "instructions": "Be brief.",
                "input": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}},
                    {"type": "function_call_output", "call_id": "call_1", "output": {"temp": 20}},
                ],
            }
        )
        messages = payload["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "Be brief."})
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": "weather?"}])
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]), {"city": "Paris"})
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(messages[3]["content"]), {"temp": 20})

    def test_function_tools_and_hosted_tools_are_accepted(self):
        payload = adapt_responses_request_to_chat(
            {
                "input": "find docs",
                "tools": [
                    {"type": "function", "name": "lookup", "description": "Lookup", "parameters": {"type": "object"}},
                    {"type": "web_search_preview", "search_context_size": "low"},
                ],
            }
        )
        tools = payload["tools"]
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "lookup")
        self.assertEqual(tools[1]["type"], "function")
        self.assertEqual(tools[1]["function"]["name"], "web_search_preview")
        self.assertIn("x_responses_hosted_tool", tools[1])

        # Existing StandardRequest builder/prompt path should consume both tools.
        standard = build_chat_standard_request(payload, default_model="gpt-3.5-turbo", surface="responses")
        self.assertIn("lookup", standard.tool_names)
        self.assertIn("web_search_preview", standard.tool_names)
        self.assertTrue(standard.tool_enabled)


class ResponsesFormatterTests(unittest.TestCase):
    def _request(self, tools=None):
        return StandardRequest(
            prompt="Human: hi\n\nAssistant:",
            response_model="gpt-4o-mini",
            resolved_model="qwen3.6-plus",
            surface="responses",
            tools=tools or [],
            tool_names=[t["name"] for t in (tools or [])],
            tool_enabled=bool(tools),
        )

    def test_non_stream_text_response_shape(self):
        execution = RuntimeExecutionResult(RuntimeAttemptState(answer_text="Hello!"), chat_id=None, acc=None)
        payload = build_responses_payload(
            response_id="resp_test",
            created_at=123,
            model_name="gpt-4o-mini",
            prompt="Human: hi",
            execution=execution,
            standard_request=self._request(),
            request_payload={"store": False},
        )
        self.assertEqual(payload["id"], "resp_test")
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["output_text"], "Hello!")
        self.assertEqual(payload["output"][0]["type"], "message")
        self.assertEqual(payload["output"][0]["content"][0]["type"], "output_text")

    def test_non_stream_tool_call_response_shape(self):
        tools = [{"name": "lookup", "description": "", "parameters": {"type": "object"}}]
        execution = RuntimeExecutionResult(
            RuntimeAttemptState(tool_calls=[{"id": "call_abc", "name": "lookup", "input": {"q": "x"}}]),
            chat_id=None,
            acc=None,
        )
        payload = build_responses_payload(
            response_id="resp_tool",
            created_at=123,
            model_name="gpt-4o-mini",
            prompt="Human: hi",
            execution=execution,
            standard_request=self._request(tools),
        )
        self.assertEqual(payload["output_text"], "")
        self.assertEqual(payload["output"][0]["type"], "function_call")
        self.assertEqual(payload["output"][0]["call_id"], "call_abc")
        self.assertEqual(payload["output"][0]["name"], "lookup")
        self.assertEqual(json.loads(payload["output"][0]["arguments"]), {"q": "x"})

    def test_stream_text_events_include_output_text_delta(self):
        translator = ResponsesStreamTranslator(response_id="resp_s", created_at=123, model_name="gpt-4o-mini")
        translator.on_delta({"phase": "answer"}, "Hel", None)
        translator.on_delta({"phase": "answer"}, "lo", None)
        execution = RuntimeExecutionResult(RuntimeAttemptState(answer_text="Hello"), chat_id=None, acc=None)
        payload = build_responses_payload(
            response_id="resp_s",
            created_at=123,
            model_name="gpt-4o-mini",
            prompt="Human: hi",
            execution=execution,
            standard_request=self._request(),
        )
        stream = "".join(translator.finalize(payload=payload))
        self.assertIn("event: response.created", stream)
        self.assertIn("event: response.output_text.delta", stream)
        self.assertIn('"delta": "Hel"', stream)
        self.assertIn("event: response.completed", stream)

    def test_stream_finalize_emits_missing_payload_tail_once(self):
        translator = ResponsesStreamTranslator(response_id="resp_tail", created_at=123, model_name="gpt-4o-mini")
        translator.on_delta({"phase": "answer"}, "已输出前半段", None)
        execution = RuntimeExecutionResult(RuntimeAttemptState(answer_text="已输出前半段，补齐尾部。"), chat_id=None, acc=None)
        payload = build_responses_payload(
            response_id="resp_tail",
            created_at=123,
            model_name="gpt-4o-mini",
            prompt="Human: hi",
            execution=execution,
            standard_request=self._request(),
        )
        stream = "".join(translator.finalize(payload=payload))
        self.assertEqual(stream.count('"delta": "，补齐尾部。"'), 1)
        self.assertIn('"text": "已输出前半段，补齐尾部。"', stream)

    def test_stream_tool_call_does_not_leak_qnml_text_delta(self):
        translator = ResponsesStreamTranslator(response_id="resp_tool_stream", created_at=123, model_name="gpt-4o-mini")
        qnml = '<|QNML|tool_calls>\n  <|QNML|invoke name="u_lookup_status">\n  </|QNML|invoke>'
        translator.on_text_chunk(qnml)
        tools = [{"name": "lookup_status", "description": "", "parameters": {"type": "object"}}]
        execution = RuntimeExecutionResult(
            RuntimeAttemptState(
                answer_text=qnml,
                tool_calls=[{"id": "toolu_123", "name": "lookup_status", "input": {"service": "qwen2api"}}],
            ),
            chat_id=None,
            acc=None,
        )
        payload = build_responses_payload(
            response_id="resp_tool_stream",
            created_at=123,
            model_name="gpt-4o-mini",
            prompt="Human: hi",
            execution=execution,
            standard_request=self._request(tools),
        )
        stream = "".join(translator.finalize(payload=payload))
        self.assertNotIn("response.output_text.delta", stream)
        self.assertNotIn("<|QNML|tool_calls>", stream)
        self.assertIn("response.function_call_arguments.done", stream)
        self.assertIn('"output_text": ""', stream)

    def test_stream_failure_events_include_terminal_completed(self):
        translator = ResponsesStreamTranslator(response_id="resp_fail", created_at=123, model_name="gpt-4o-mini")
        stream = "".join(translator.fail(error={"code": "upstream_error", "message": "boom"}))
        self.assertIn("event: response.failed", stream)
        self.assertIn('"status": "failed"', stream)
        self.assertIn('"code": "upstream_error"', stream)
        self.assertIn("event: response.completed", stream)
        self.assertTrue(stream.endswith("data: [DONE]\n\n"))

    def test_stream_tool_call_events_include_arguments_done(self):
        translator = ResponsesStreamTranslator(response_id="resp_s", created_at=123, model_name="gpt-4o-mini")
        translator.start()
        tools = [{"name": "lookup", "description": "", "parameters": {"type": "object"}}]
        execution = RuntimeExecutionResult(
            RuntimeAttemptState(tool_calls=[{"id": "call_abc", "name": "lookup", "input": {"q": "x"}}]),
            chat_id=None,
            acc=None,
        )
        payload = build_responses_payload(
            response_id="resp_s",
            created_at=123,
            model_name="gpt-4o-mini",
            prompt="Human: hi",
            execution=execution,
            standard_request=self._request(tools),
        )
        stream = "".join(translator.finalize(payload=payload))
        self.assertIn("event: response.function_call_arguments.done", stream)
        self.assertIn('"arguments": "{\\"q\\": \\"x\\"}"', stream)
        self.assertIn("event: response.completed", stream)


if __name__ == "__main__":
    unittest.main()

class ToolSchemaCoercionTests(unittest.TestCase):
    def test_string_array_field_split_for_checks(self):
        from backend.services import tool_parser
        tools = [{
            "name": "lookup_status",
            "parameters": {
                "type": "object",
                "properties": {"checks": {"type": "array", "items": {"type": "string"}}},
                "required": ["checks"],
            },
        }]
        blocks, stop = tool_parser.parse_tool_calls_silent(
            '<|QNML|tool_calls>\n'
            '  <|QNML|invoke name="u_lookup_status">\n'
            '    <|QNML|parameter name="checks">healthzresponses_streamlogs</|QNML|parameter>\n'
            '  </|QNML|invoke>\n',
            tools,
        )
        self.assertEqual(stop, "tool_use")
        call = next(b for b in blocks if b.get("type") == "tool_use")
        self.assertEqual(call["input"]["checks"], ["responses_stream", "healthz", "logs"])

class MaxOutputTokenMappingTests(unittest.TestCase):
    def test_responses_max_output_tokens_reaches_standard_request_and_payload(self):
        from backend.services.responses_adapter import adapt_responses_request_to_chat
        from backend.services.standard_request_builder import build_chat_standard_request
        from backend.upstream.payload_builder import build_chat_payload

        adapted = adapt_responses_request_to_chat({
            "model": "gpt-5",
            "input": "write a long answer",
            "max_output_tokens": "23",
            "reasoning": {"effort": "low"},
        })
        standard = build_chat_standard_request(adapted, default_model="gpt-3.5-turbo", surface="responses")
        self.assertEqual(standard.max_output_tokens, 23)
        self.assertEqual(standard.reasoning_effort, "low")

        payload = build_chat_payload(
            "chat_1",
            standard.resolved_model,
            standard.prompt,
            thinking_enabled=standard.thinking_enabled,
            reasoning_effort=standard.reasoning_effort,
            max_output_tokens=standard.max_output_tokens,
        )
        feature_config = payload["messages"][0]["feature_config"]
        self.assertEqual(payload["max_output_tokens"], 23)
        self.assertEqual(payload["max_tokens"], 23)
        self.assertEqual(payload["max_new_tokens"], 23)
        self.assertEqual(feature_config["max_output_tokens"], 23)
        self.assertEqual(feature_config["max_tokens"], 23)
        self.assertEqual(feature_config["thinking_mode"], "Disabled")

class ResponsesIncompletePayloadTests(unittest.TestCase):
    def test_incomplete_reason_sets_responses_status(self):
        tools = []
        state = RuntimeAttemptState(answer_text="partial **")
        state.incomplete_reason = "max_output_tokens"
        execution = RuntimeExecutionResult(state, chat_id=None, acc=None)
        payload = build_responses_payload(
            response_id="resp_incomplete",
            created_at=123,
            model_name="gpt-4o-mini",
            prompt="Human: hi",
            execution=execution,
            standard_request=ResponsesFormatterTests()._request(tools),
            request_payload={"max_output_tokens": 30},
        )
        self.assertEqual(payload["status"], "incomplete")
        self.assertEqual(payload["incomplete_details"], {"reason": "max_output_tokens"})
