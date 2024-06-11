import logging
from typing import Any, AsyncGenerator, List, Optional, Tuple

import anthropic
from langchain.chains import ConversationChain
from langchain.chat_models import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import AIMessage, ChatMessage, HumanMessage

from vocode import getenv
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.agent.utils import (
    anthropic_get_tokens,
    collate_response_async,
    format_openai_chat_messages_from_transcript,
    get_sentence_from_buffer,
    openai_get_tokens,
)
from vocode.streaming.models.agent import ChatAnthropicAgentConfig
from vocode.streaming.models.message import BaseMessage

SENTENCE_ENDINGS = [".", "!", "?"]


class ChatAnthropicAgent(RespondAgent[ChatAnthropicAgentConfig]):
    def __init__(
        self,
        agent_config: ChatAnthropicAgentConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)
        anthropic_api_key = getenv("ANTHROPIC_API_KEY")

        if not anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in environment or passed in"
            )

        self.anthropic_async_client = anthropic.AsyncAnthropic(
            api_key=anthropic_api_key, max_retries=2
        )

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[BaseMessage, bool]:
        text = await self.conversation.apredict(input=human_input)
        self.logger.debug(f"LLM response: {text}")
        return BaseMessage(text=text), False

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[BaseMessage, None]:
        ### We should implement get_cut_off_response() for Anthropic??
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            yield BaseMessage(text=cut_off_response)
            return
        assert self.transcript is not None

        messages_formatted = format_openai_chat_messages_from_transcript(
            self.transcript
        )

        stream = await self.anthropic_async_client.messages.create(
            max_tokens=1024,
            messages=messages_formatted,
            system=self.agent_config.prompt_preamble,
            model="claude-3-opus-20240229",
            stream=True,
        )

        async for message in collate_response_async(
            anthropic_get_tokens(stream), get_functions=True
        ):
            yield BaseMessage(text=message)

    def update_last_bot_message_on_cut_off(self, message: str):
        for memory_message in self.memory.chat_memory.messages[::-1]:
            if (
                isinstance(memory_message, ChatMessage)
                and memory_message.role == "assistant"
            ) or isinstance(memory_message, AIMessage):
                memory_message.content = message
                return
