import asyncio
from backend.services.mock_upstream import MockUpstream

async def simulate_mock_stream(translator, mock: MockUpstream, delay: float = 0.15):
    """If mock is in stream_mode, feed chunks one by one to translator.on_text_chunk with delay."""
    if not getattr(mock, 'stream_mode', False) or not getattr(mock, 'stream_chunks', None):
        return False
    
    for chunk in mock.stream_chunks:
        if chunk:
            translator.on_text_chunk(chunk)
            await asyncio.sleep(delay)
    return True
