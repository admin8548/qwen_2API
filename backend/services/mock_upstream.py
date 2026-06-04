from backend.runtime.execution import RuntimeExecutionResult as ExecutionResult, RuntimeAttemptState as RuntimeExecutionState
import asyncio


class MockUpstream:
    """Mock upstream for development and testing of /v1/responses.
    Returns controllable responses without calling real Qwen service.
    Now supports simulated streaming via chunked generation.
    """
    
    def __init__(self):
        self.default_reply = "你好！我是基于 Qwen 的 AI 助手，很高兴为你服务。我可以帮助你回答问题、编写代码、分析文档等。"
        self.stream_mode = False
        self.stream_chunks = None
    
    async def generate(self, standard_request, **kwargs):
        """Return a mock execution result. Supports basic streaming simulation."""
        prompt = standard_request.prompt or standard_request.full_prompt or "Hello"
        
        state = RuntimeExecutionState()
        state.answer_text = self.default_reply
        
        # If stream_mode is enabled, we can later hook into chunking in the bridge
        if self.stream_mode and self.stream_chunks:
            state.stream_chunks = self.stream_chunks  # will be consumed by updated bridge if present
        
        return ExecutionResult(
            execution=type('obj', (object,), {'state': state})(),
            prompt=prompt,
            directive=None,
            usage={"prompt_tokens": len(prompt)//4, "completion_tokens": 30, "total_tokens": 50}
        )
    
    def set_reply(self, text: str):
        """Allow dynamic reply for testing."""
        self.default_reply = text
        self.stream_mode = False
        self.stream_chunks = None
    
    def set_stream_reply(self, chunks: list[str]):
        """Set reply as list of text chunks for simulated streaming."""
        self.default_reply = "".join(chunks)
        self.stream_mode = True
        self.stream_chunks = chunks
