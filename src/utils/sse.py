"""Decode SSE independently of HTTP byte boundaries (including UTF-8)."""

import codecs
from collections.abc import AsyncIterator


async def iter_sse_data(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    data = []
    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if data:
                    yield "\n".join(data)
                    data.clear()
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
            # Ignore SSE comments, event, id and retry fields.
    buffer += decoder.decode(b"", final=True)
    if buffer.startswith("data:"):
        value = buffer[5:].removesuffix("\r")
        data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)
