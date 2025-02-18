import re
from copy import deepcopy
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from vocode.streaming.models.actions import FunctionCall, FunctionFragment
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    ActionFinish,
    ActionStart,
    EventLog,
    Message,
    Transcript,
)

SENTENCE_ENDINGS = [".", "!", "?", "\n"]

# Ideas to fool this regex:
# Abbreviations longer than 4 characters, such as Msss. or Corp.

def mark_commands(text: str) -> str:
    """Replace command content in brackets with space so it won't trigger sentence boundaries"""
    def replace_match(match):
        # Replace command content with equivalent number of underscores
        return ' ' * len(match.group(0))
    
    # Match [...] patterns, handle nested brackets by being non-greedy
    command_pattern = re.compile(r'\[.*?\]')
    return command_pattern.sub(replace_match, text)

SENTENCE_BOUNDARY = re.compile(
    r"""
    [\r\n]+\s*  # Any number of newlines automatically is a boundary
    |
    [。！？]\s* # CJK sentence endings always indicate a boundary
    |
    \S{4,}[“"'.!?]\s+(?=[A-ZÖÄÅ]) # A sentence of 4+ non-whitespace chars ending with punctuation and next starting with capital letter
                                    # This should mean short abbreviations and list items don't get split
""", re.VERBOSE)

def find_sentence_boundary(buffer: str) -> int:
    marked_buffer = mark_commands(buffer)

    match = SENTENCE_BOUNDARY.search(marked_buffer)
    if match:
        return match.end()
    return -1

async def collate_response_async(
    gen: AsyncIterable[Union[str, FunctionFragment]],
    get_functions: Literal[True, False] = False,
) -> AsyncGenerator[Union[str, FunctionCall], None]:
    buffer = ""
    function_name_buffer = ""
    function_args_buffer = ""
    async for token in gen:
        if not token:
            continue
        if isinstance(token, str):
            buffer += token
            pos = find_sentence_boundary(buffer)
            if pos > 0:
                sentence = buffer[:pos].strip()
                yield sentence
                buffer = buffer[pos:]
        elif isinstance(token, FunctionFragment):
            function_name_buffer += token.name
            function_args_buffer += token.arguments
    to_return = buffer.strip()
    if to_return:
        yield to_return
    if function_name_buffer and get_functions:
        yield FunctionCall(name=function_name_buffer, arguments=function_args_buffer)

async def openai_get_tokens(gen) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    async for chunk in gen:
        # When requesting usage data for streaming, it will come as an extra final token
        # We yield this in form of bracketed commands that has to be parsed (and removed)
        # downstream
        if chunk.usage and chunk.usage.prompt_tokens:
            yield f"[prompt_tokens: {chunk.usage.prompt_tokens}]"
        if chunk.usage and chunk.usage.completion_tokens:
            yield f"[completion_tokens: {chunk.usage.completion_tokens}]"
        choices = chunk.choices or []        
        if len(choices) == 0:
            break
        choice = choices[0]
        if choice.finish_reason:
            yield f"[finish_reason: {choice.finish_reason}]"
            continue
        delta = choice.delta or {}
        if hasattr(delta, "text") and getattr(delta, "text"):
            token = delta.text
            yield token
        if hasattr(delta, "content") and getattr(delta, "content"):
            token = delta.content
            yield token
        elif hasattr(delta, "tool_calls") and getattr(delta, "tool_calls"):
            yield FunctionFragment(
                name=delta.tool_calls[0].function.name
                if (hasattr(delta.tool_calls[0].function, "name") and getattr(delta.tool_calls[0].function, "name"))
                else "",
                arguments=delta.tool_calls[0].function.arguments
                if (hasattr(delta.tool_calls[0].function, "arguments") and getattr(delta.tool_calls[0].function, "arguments"))
                else "",
            )


async def anthropic_get_tokens(
    gen,
) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    async for event in gen:
        if event.type == "content_block_stop":
            break
        if event.type == "content_block_start":
            token = event.content_block.text
            yield token
        if event.type == "content_block_delta":
            delta = event.delta
            if hasattr(delta, "text") and getattr(delta, "text"):
                token = delta.text
                token = token.lstrip("\n")
                yield token


def find_last_punctuation(buffer: str) -> Optional[int]:
    indices = [buffer.rfind(ending) for ending in SENTENCE_ENDINGS]
    if not indices:
        return None
    return max(indices)


def get_sentence_from_buffer(buffer: str):
    last_punctuation = find_last_punctuation(buffer)
    if last_punctuation:
        return buffer[: last_punctuation + 1], buffer[last_punctuation + 1 :]
    else:
        return None, None


def format_openai_chat_messages_from_transcript(
    transcript: Transcript, prompt_preamble: Optional[str] = None
) -> List[dict]:
    chat_messages: List[Dict[str, Optional[Any]]] = (
        [{"role": "system", "content": prompt_preamble}] if prompt_preamble else []
    )

    # merge consecutive bot messages
    new_event_logs: List[EventLog] = []
    idx = 0
    while idx < len(transcript.event_logs):
        bot_messages_buffer: List[Message] = []
        current_log = transcript.event_logs[idx]
        while isinstance(current_log, Message) and current_log.sender == Sender.BOT:
            bot_messages_buffer.append(current_log)
            idx += 1
            try:
                current_log = transcript.event_logs[idx]
            except IndexError:
                break
        if bot_messages_buffer:
            merged_bot_message = deepcopy(bot_messages_buffer[-1])
            merged_bot_message.text = " ".join(
                event_log.text for event_log in bot_messages_buffer
            )
            new_event_logs.append(merged_bot_message)
        else:
            new_event_logs.append(current_log)
            idx += 1

    for event_log in new_event_logs:
        if isinstance(event_log, Message):
            chat_messages.append(
                {
                    "role": "assistant" if event_log.sender == Sender.BOT else "user",
                    "content": event_log.text,
                }
            )
        elif isinstance(event_log, ActionStart):
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": event_log.action_type,
                        "arguments": event_log.action_input.params.json(),
                    },
                }
            )
        elif isinstance(event_log, ActionFinish):
            chat_messages.append(
                {
                    "role": "function",
                    "name": event_log.action_type,
                    "content": event_log.action_output.response.json(),
                }
            )
    return chat_messages


def vector_db_result_to_openai_chat_message(vector_db_result):
    return {"role": "user", "content": vector_db_result}
