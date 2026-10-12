"""Real streaming interface shape, entirely in memory."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


class Stream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self):
        self.closed = True


def chunk(content='', reasoning='', finish=None, usage=None):
    return SimpleNamespace(model='resolved-model', usage=SimpleNamespace(model_dump=lambda: usage) if usage else None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, reasoning_content=reasoning), finish_reason=finish)])


def sdk(raw='Response', finish='stop', usage=None, chunks=None, capability=None):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value={'models': capability or []})
    stream = Stream(chunks if chunks is not None else [chunk(raw, finish=finish, usage=usage)])
    client.stream = stream
    client.chat.completions.create = AsyncMock(return_value=stream)
    return client
