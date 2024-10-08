import asyncio
import logging
from typing import Any, AsyncGenerator, Optional, Tuple, Union
import wave
import aiohttp
from opentelemetry.trace import Span

from vocode import getenv
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
    tracer,
)
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils.mp3_helper import decode_mp3
from vocode.streaming.synthesizer.miniaudio_worker import MiniaudioWorker
from vocode.streaming.synthesizer.ovrlipsync.ovrlipsync import OVRLipsyncProcessor
from vocode.streaming.utils import get_chunk_size_per_second



ADAM_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io/v1/"
lipsync_processor = OVRLipsyncProcessor(24000) # TODO currently hardcoded

def get_lipsync_events(from_s: int, to_s: int, lipsync_events: list) -> list:
    events = [
    {
        "audio_offset": event['audio_offset'] - from_s,
        "viseme_id": event["viseme_id"],
    }
     for event in lipsync_events if event["audio_offset"] and from_s <= event["audio_offset"] < to_s]
    return events

class ElevenLabsSynthesizer(BaseSynthesizer[ElevenLabsSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: ElevenLabsSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        import elevenlabs

        self.elevenlabs = elevenlabs

        self.api_key = synthesizer_config.api_key or getenv("ELEVEN_LABS_API_KEY")
        self.voice_id = synthesizer_config.voice_id or ADAM_VOICE_ID
        self.stability = synthesizer_config.stability
        self.similarity_boost = synthesizer_config.similarity_boost
        self.model_id = synthesizer_config.model_id
        self.optimize_streaming_latency = synthesizer_config.optimize_streaming_latency
        self.words_per_minute = 150
        self.experimental_streaming = synthesizer_config.experimental_streaming
        self.logger = logger
        self.bytes_per_second = get_chunk_size_per_second(synthesizer_config.audio_encoding, synthesizer_config.sampling_rate)

        if lipsync_processor.sample_rate == synthesizer_config.sampling_rate:
            self.lipsync_processor = lipsync_processor
        else:  
            self.logger.warning(f"OVRLipsyncProcessor not started because sample rate {lipsync_processor.sample_rate} does not match synthesizer sample rate {synthesizer_config.sampling_rate}")

    async def experimental_streaming_output_generator(
        self,
        response: aiohttp.ClientResponse,
        chunk_size: int,
        create_speech_span: Optional[Span],
        lipsync_events: Optional[list] = None,
    ) -> AsyncGenerator[SynthesisResult.ChunkResult, None]:
        miniaudio_worker_input_queue: asyncio.Queue[
            Union[bytes, None]
        ] = asyncio.Queue()
        miniaudio_worker_output_queue: asyncio.Queue[
            Tuple[bytes, bool]
        ] = asyncio.Queue()
        miniaudio_worker = MiniaudioWorker(
            self.synthesizer_config,
            chunk_size,
            miniaudio_worker_input_queue,
            miniaudio_worker_output_queue,
        )
        miniaudio_worker.start()
        stream_reader = response.content

        # Create a task to send the mp3 chunks to the MiniaudioWorker's input queue in a separate loop
        async def send_chunks():
            async for chunk in stream_reader.iter_any():
                miniaudio_worker.consume_nonblocking(chunk)
            miniaudio_worker.consume_nonblocking(None)  # sentinel

        try:
            asyncio.create_task(send_chunks())
            audio_offset = 0.0
            # Await the output queue of the MiniaudioWorker and yield the wav chunks in another loop
            while True:
                # Get the wav chunk and the flag from the output queue of the MiniaudioWorker
                wav_chunk, is_last = await miniaudio_worker.output_queue.get()
                if lipsync_events is not None and self.lipsync_processor:
                    if not lipsync_processor.process:
                        await lipsync_processor.start()
                    lipsync_in_chunk = await self.lipsync_processor.detect_lipsync(wav_chunk, audio_offset)
                    lipsync_events.extend(lipsync_in_chunk)
                    audio_offset += len(wav_chunk) / self.bytes_per_second

                if self.synthesizer_config.should_encode_as_wav:
                    wav_chunk = encode_as_wav(wav_chunk, self.synthesizer_config)

                yield SynthesisResult.ChunkResult(wav_chunk, is_last)
                # If this is the last chunk, break the loop
                if is_last and create_speech_span is not None:
                    create_speech_span.end()
                    break
        except asyncio.CancelledError:
            pass
        finally:
            miniaudio_worker.terminate()

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        voice = self.elevenlabs.Voice(voice_id=self.voice_id)
        if self.stability is not None and self.similarity_boost is not None:
            voice.settings = self.elevenlabs.VoiceSettings(
                stability=self.stability, similarity_boost=self.similarity_boost
            )
        url = ELEVEN_LABS_BASE_URL + f"text-to-speech/{self.voice_id}"

        if self.experimental_streaming:
            url += "/stream"

        if self.optimize_streaming_latency:
            url += f"?optimize_streaming_latency={self.optimize_streaming_latency}"
        headers = {"xi-api-key": self.api_key}
        body = {
            "text": message.text,
            "voice_settings": voice.settings.dict() if voice.settings else None,
        }
        if self.model_id:
            body["model_id"] = self.model_id

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.create_total",
        )

        session = self.aiohttp_session

        max_attempts = 3
        delay = 0.5
        backoff = 1.8

        attempts = 0
        response = None
        while attempts < max_attempts:
            try:
                response = await session.request(
                    "POST",
                    url,
                    json=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=9),
                )
                if response.ok:
                    break
                elif response.status == 429:
                    self.logger.warning("ElevenLabs' rate limit exceeded. Retrying after delay...")
                    await asyncio.sleep(delay)
                    delay = delay * backoff  # Increase delay for next attempt
                    attempts += 1
                else:
                    Exception(f"ElevenLabs API returned {response.status} status code")
            except asyncio.TimeoutError:
                self.logger.warning("ElevenLabs timed out. Retrying after delay...")
                await asyncio.sleep(delay)
                delay = delay * backoff  # Increase delay for next attempt
                attempts += 1
            except Exception as e:
                raise e

        if not response or not response.ok:
            raise Exception(f"Failed to retrieve response from ElevenLabs after {attempts} tries for '{message.text}'.")

        if self.experimental_streaming:
            lipsync_events = []
            return SynthesisResult(
                self.experimental_streaming_output_generator(
                    response, chunk_size, create_speech_span, lipsync_events
                ),  # should be wav
                lambda seconds: self.get_message_cutoff_from_voice_speed(
                    message, seconds, self.words_per_minute
                ),
                lambda from_s, to_s: get_lipsync_events(from_s, to_s, lipsync_events),
            )
        else:
            audio_data = await response.read()
            create_speech_span.end()
            convert_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.ELEVEN_LABS.value.split('_', 1)[-1]}.convert",
            )
            # Decodes MP3 into WAV
            output_bytes_io = decode_mp3(audio_data)

            result = self.create_synthesis_result_from_wav(
                file=output_bytes_io,
                message=message,
                chunk_size=chunk_size,
            )
            convert_span.end()

            return result
