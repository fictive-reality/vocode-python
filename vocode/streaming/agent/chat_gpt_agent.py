import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import openai

from vocode import getenv
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.agent.utils import (
    collate_response_async,
    format_openai_chat_messages_from_transcript,
    openai_get_tokens,
    vector_db_result_to_openai_chat_message,
)
from vocode.streaming.models.actions import FunctionCall
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcript import Transcript
from vocode.streaming.vector_db.factory import VectorDBFactory

please_repeat = {
    "en": [
        "Sorry I think I didn't get that, could you repeat it please?",
        "I'm sorry, I didn't catch that, could you say it again?",
        "Pardon me, I didn't understand, can you repeat that?",
    ],
    "sv": [
        "Förlåt, jag tror att jag inte fick det, kan du upprepa det tack?",
        "Jag är ledsen, jag hörde inte, kan du säga det igen?",
        "Ursäkta mig, jag förstod inte, kan du upprepa det?",
    ],
    "es": [
        "Perdón, creo que no entendí eso, ¿podrías repetirlo por favor?",
        "Lo siento, creo que no he entendido del todo a que te referías, ¿podrías decirlo otra vez?",
        "Disculpa, no comprendí, ¿puedes repetir eso?",
    ],
    "de": [
        "Entschuldigung, ich glaube, ich habe das nicht verstanden, können Sie das bitte wiederholen?",
        "Es tut mir leid, ich habe das nicht verstanden, können Sie es noch einmal sagen?",
        "Verzeihung, ich habe das nicht verstanden, können Sie das wiederholen?",
    ],
    "fi": [
        "Anteeksi, luulen etten ymmärtänyt, voitko toistaa sen, kiitos?",
        "Olen pahoillani, en saanut siitä selvää, voitko sanoa sen uudelleen?",
        "Anteeksi, en ymmärtänyt, voitko toistaa sen?",
    ],
    "fr": [
        "Désolé, je pense que je n'ai pas compris, pouvez-vous répéter s'il vous plaît?",
        "Je suis désolé, je n'ai pas compris, pouvez-vous le dire encore une fois?",
        "Pardon, je n'ai pas compris, pouvez-vous répéter?",
    ],
    "ar": [
        "آسف، أعتقد أنني لم أفهم ذلك، هل يمكنك تكراره من فضلك؟",
        "أنا آسف، لم أفهم ذلك، هل يمكنك أن تقوله مرة أخرى؟",
        "عذرًا، لم أفهم، هل يمكنك تكرار ذلك؟",
    ],
    "zh": [
        "对不起，我想我没听懂，能请你再说一遍吗？",
        "对不起，我没听清楚，能再说一遍吗？",
        "抱歉，我没明白，你能重复一遍吗？",
    ],
    "hu": [
        "Sajnálom, azt hiszem, nem értettem, meg tudná ismételni kérem?",
        "Elnézést, nem értettem, el tudná mondani újra?",
        "Bocsánat, nem értettem, meg tudná ismételni?",
    ],
    "nl": [
        "Sorry, ik denk dat ik het niet begreep, kun je het herhalen alstublieft?",
        "Het spijt me, ik heb het niet verstaan, kun je het nog een keer zeggen?",
        "Pardon, ik heb het niet begrepen, kun je dat herhalen?",
    ],
    "ru": [
        "Извините, я думаю, я не понял, не могли бы вы повторить, пожалуйста?",
        "Прошу прощения, я не понял, не могли бы вы сказать это снова?",
        "Извините, я не понял, не могли бы вы повторить это?",
    ],
    "fa": [
        "ببخشید فکر کنم متوجه نشدم، میشه لطفاً تکرارش کنید؟",
        "متاسفم، متوجه نشدم، میشه دوباره بگید؟",
        "عذر می‌خواهم، نفهمیدم، میشه تکرار کنید؟",
    ],
    "hi": [
        "मुझे खेद है, मुझे यह समझ में नहीं आया, क्या आप कृपया इसे दोहरा सकते हैं?",
        "मुझे माफ करें, मुझे यह समझ में नहीं आया, क्या आप इसे फिर से कह सकते हैं?",
        "माफ करें, मैंने समझा नहीं, क्या आप इसे दोहरा सकते हैं?",
    ],
    "ur": [
        "معاف کیجیے گا، میرا خیال ہے کہ میں نے سمجھا نہیں، کیا آپ براہِ کرم اسے دہرا سکتے ہیں؟",
        "مجھے معاف کریں، میں نے یہ نہیں سمجھا، کیا آپ اسے دوبارہ کہہ سکتے ہیں؟",
        "معذرت، میں سمجھا نہیں، کیا آپ اسے دہرا سکتے ہیں؟",
    ],
    "pa": [
        "ਮਾਫ ਕਰਨਾ, ਮੈਨੂੰ ਲੱਗਦਾ ਹੈ ਕਿ ਮੈਨੂੰ ਇਹ ਸਮਝ ਨਹੀਂ ਆਇਆ, ਕੀ ਤੁਸੀਂ ਕਿਰਪਾ ਕਰਕੇ ਇਸਨੂੰ ਦੁਹਰਾ ਸਕਦੇ ਹੋ?",
        "ਮੈਨੂੰ ਮਾਫ ਕਰੋ, ਮੈਂ ਇਹ ਸਮਝ ਨਹੀਂ ਆਇਆ, ਕੀ ਤੁਸੀਂ ਇਸਨੂੰ ਫਿਰ ਕਹਿ ਸਕਦੇ ਹੋ?",
        "ਮਾਫ ਕਰਨਾ, ਮੈਂ ਸਮਝ ਨਹੀਂ ਆਇਆ, ਕੀ ਤੁਸੀਂ ਇਸਨੂੰ ਦੁਹਰਾ ਸਕਦੇ ਹੋ?",
    ],
    "pt": [
        "Desculpe, acho que não entendi, você pode repetir, por favor?",
        "Desculpe, não entendi, você pode dizer isso de novo?",
        "Perdão, não entendi, você pode repetir isso?",
    ],
    "pl": [
        "Przepraszam, myślę, że nie zrozumiałem, czy możesz to powtórzyć, proszę?",
        "Przepraszam, nie złapałem tego, możesz to powiedzieć jeszcze raz?",
        "Przepraszam, nie zrozumiałem, czy możesz to powtórzyć?",
    ],
    "bn": [
        "দুঃখিত, আমি বোধহয় এটা বুঝতে পারিনি, আপনি কি দয়া করে আবার বলতে পারেন?",
        "আমাকে দুঃখিত, আমি বুঝতে পারিনি, আপনি কি আবার বলতে পারেন?",
        "মাফ করবেন, আমি বুঝতে পারিনি, আপনি কি এটা পুনরায় বলতে পারেন?",
    ],
    "te": [
        "క్షమించండి, నేను దాన్ని అర్థం చేసుకోలేదని అనుకుంటున్నాను, దయచేసి మళ్ళీ చెప్తారా?",
        "క్షమించాలి, నేను అర్థం చేసుకోలేకపోయాను, మీరు దానిని మళ్ళీ చెప్పగలరా?",
        "నన్ను క్షమించండి, నాకు అర్థం కాలేదు, దయచేసి మీరు దాన్ని పునరావృతం చేయగలరా?",
    ],
}


class ChatGPTAgent(RespondAgent[ChatGPTAgentConfig]):
    def __init__(
        self,
        agent_config: ChatGPTAgentConfig,
        action_factory: ActionFactory = ActionFactory(),
        logger: Optional[logging.Logger] = None,
        openai_api_key: Optional[str] = None,
        vector_db_factory=VectorDBFactory(),
    ):
        super().__init__(
            agent_config=agent_config, action_factory=action_factory, logger=logger
        )
        if agent_config.azure_params:
            openai.api_type = agent_config.azure_params.api_type
            openai.api_version = agent_config.azure_params.api_version
            openai.api_key = getenv("AZURE_OPENAI_API_KEY")
            self.openai_client = openai.AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-05-15",
            )
            self.openai_async_client = openai.AsyncAzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-05-15",
            )
        else:
            openai.api_type = "open_ai"
            openai.api_version = None
            openai.api_key = openai_api_key or getenv("OPENAI_API_KEY")
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.openai_async_client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        self.first_response = (
            self.create_first_response(agent_config.expected_first_prompt)
            if agent_config.expected_first_prompt
            else None
        )
        self.is_first_response = True

        if self.agent_config.vector_db_config:
            self.vector_db = vector_db_factory.create_vector_db(
                self.agent_config.vector_db_config
            )

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
        ]

    def get_chat_parameters(self, messages: Optional[List] = None):
        assert self.transcript is not None
        messages = messages or format_openai_chat_messages_from_transcript(
            self.transcript, self.agent_config.prompt_preamble
        )

        parameters: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.agent_config.max_tokens,
            "temperature": self.agent_config.temperature,
        }

        if self.agent_config.azure_params is not None:
            parameters["model"] = self.agent_config.azure_params.engine
        else:
            parameters["model"] = self.agent_config.model_name

        if self.functions:
            parameters["functions"] = self.functions

        return parameters

    async def create_first_response(self, first_prompt):
        messages = [
            (
                [{"role": "system", "content": self.agent_config.prompt_preamble}]
                if self.agent_config.prompt_preamble
                else []
            )
            + [{"role": "user", "content": first_prompt}]
        ]

        parameters = self.get_chat_parameters(messages)
        response = await self.openai_async_client.chat.completions.create(**parameters)
        return response.choices[0].content

    def attach_transcript(self, transcript: Transcript):
        self.transcript = transcript

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[BaseMessage, bool]:
        assert self.transcript is not None
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            return BaseMessage(text=cut_off_response), False
        self.logger.debug("LLM responding to human input")
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            text = self.first_response
        else:
            chat_parameters = self.get_chat_parameters()
            chat_completion = await self.openai_async_client.chat.completions.create(
                **chat_parameters
            )
            text = chat_completion.choices[0].content
        self.logger.debug(f"LLM response: {text}")
        return BaseMessage(text=text), False

    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[Union[BaseMessage, FunctionCall], None]:
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            yield BaseMessage(text=cut_off_response)
            return
        assert self.transcript is not None

        if self.agent_config.vector_db_config:
            docs_with_scores = await self.vector_db.similarity_search_with_score(
                self.transcript.get_last_user_message()[1]
            )
            docs_with_scores_str = "\n\n".join(
                [
                    "Document: "
                    + doc[0].metadata["source"]
                    + f" (Confidence: {doc[1]})\n"
                    + doc[0].lc_kwargs["page_content"].replace(r"\n", "\n")
                    for doc in docs_with_scores
                ]
            )
            vector_db_result = f"Found {len(docs_with_scores)} similar documents:\n{docs_with_scores_str}"
            messages = format_openai_chat_messages_from_transcript(
                self.transcript, self.agent_config.prompt_preamble
            )
            messages.insert(
                -1, vector_db_result_to_openai_chat_message(vector_db_result)
            )
            chat_parameters = self.get_chat_parameters(messages)
        else:
            chat_parameters = self.get_chat_parameters()
        chat_parameters["stream"] = True

        try:
            stream = await self.openai_async_client.chat.completions.create(
                **chat_parameters
            )
        except openai.BadRequestError as e:
            if e.status_code == 400:
                self.logger.warning(
                    "OpenAI returned error code 400 on generate response. Continuing conversation with an empty response..."
                )
                yield BaseMessage(text=please_repeat[self.agent_config.locale or "en"])
                return
            else:
                self.logger.exception("An unexpected error occured: ")
                raise

        async for message in collate_response_async(
            openai_get_tokens(stream), get_functions=True
        ):
            yield BaseMessage(text=message)
