"""
Stream-aware PII reverter.
Buffers tokens to ensure PII placeholders (e.g., <ACCOUNT_NUMBER_1>) 
are not partially streamed to the user before they can be replaced.
"""
from typing import AsyncGenerator

async def stream_restore(stream: AsyncGenerator[str, None], mapping: dict[str, str]) -> AsyncGenerator[str, None]:
    """
    Consumes a stream of tokens, buffers any potential PII placeholders,
    replaces them if fully formed, and yields the safe text.
    """
    buffer = ""
    async for token in stream:
        buffer += token
        
        # If there's a `<` in the buffer, we might be inside a placeholder.
        if "<" in buffer:
            last_open = buffer.rfind("<")
            last_close = buffer.rfind(">")
            
            # If the last `<` is NOT closed by a subsequent `>`, hold everything from `<` onwards.
            if last_open > last_close:
                safe_to_yield = buffer[:last_open]
                buffer = buffer[last_open:]
            else:
                # All opened tags are closed
                safe_to_yield = buffer
                buffer = ""
        else:
            safe_to_yield = buffer
            buffer = ""
            
        if safe_to_yield:
            # Replace any complete placeholders
            for placeholder, original in mapping.items():
                safe_to_yield = safe_to_yield.replace(placeholder, original)
            yield safe_to_yield
            
    # Flush remaining buffer
    if buffer:
        for placeholder, original in mapping.items():
            buffer = buffer.replace(placeholder, original)
        yield buffer
