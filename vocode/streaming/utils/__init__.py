import asyncio
import audioop
import logging
import os
import secrets
from typing import Any
import wave
from string import ascii_letters, digits

from ..models.audio_encoding import AudioEncoding

logger = logging.getLogger(__name__)
custom_alphabet = ascii_letters + digits + ".-_"

ALL_VISEME_LABELS = {
    # Azure
    "azure_0": "silence",
    "azure_1": "æ, ə, ʌ",
    "azure_2": "ɑ",
    "azure_3": "ɔ",
    "azure_4": "ɛ, ʊ",
    "azure_5": "ɝ",
    "azure_6": "j, i, ɪ",
    "azure_7": "w, u",
    "azure_8": "o",
    "azure_9": "aʊ",
    "azure_10": "ɔɪ",
    "azure_11": "aɪ",
    "azure_12": "h",
    "azure_13": "ɹ",
    "azure_14": "l",
    "azure_15": "s, z",
    "azure_16": "ʃ, tʃ, dʒ, ʒ",
    "azure_17": "ð",
    "azure_18": "f, v",
    "azure_19": "d, t, n, θ",
    "azure_20": "k, g, ŋ",
    "azure_21": "p, b, m",
    # OVR
    "ovr_sil": "sil",
    "ovr_PP": "PP",
    "ovr_FF": "FF",
    "ovr_TH": "TH",
    "ovr_DD": "DD",
    "ovr_kk": "kk",
    "ovr_CH": "CH",
    "ovr_SS": "SS",
    "ovr_nn": "nn",
    "ovr_RR": "RR",
    "ovr_aa": "aa",
    "ovr_E": "E",
    "ovr_I": "I",
    "ovr_O": "O",
    "ovr_U": "U",
}

def create_loop_in_thread(loop: asyncio.AbstractEventLoop, long_running_task=None):
    asyncio.set_event_loop(loop)
    if long_running_task:
        loop.run_until_complete(long_running_task)
    else:
        loop.run_forever()


def convert_linear_audio(
    raw_wav: bytes,
    input_sample_rate=24000,
    output_sample_rate=8000,
    output_encoding=AudioEncoding.LINEAR16,
    output_sample_width=2,
):
    # downsample
    if input_sample_rate != output_sample_rate:
        raw_wav, _ = audioop.ratecv(
            raw_wav, 2, 1, input_sample_rate, output_sample_rate, None
        )

    if output_encoding == AudioEncoding.LINEAR16:
        return raw_wav
    elif output_encoding == AudioEncoding.MULAW:
        return audioop.lin2ulaw(raw_wav, output_sample_width)


def convert_wav(
    file: Any,
    output_sample_rate=8000,
    output_encoding=AudioEncoding.LINEAR16,
):
    with wave.open(file, "rb") as wav:
        raw_wav = wav.readframes(wav.getnframes())
        return convert_linear_audio(
            raw_wav,
            input_sample_rate=wav.getframerate(),
            output_sample_rate=output_sample_rate,
            output_encoding=output_encoding,
            output_sample_width=wav.getsampwidth(),
        )


def get_chunk_size_per_second(audio_encoding: AudioEncoding, sampling_rate: int) -> int:
    if audio_encoding == AudioEncoding.LINEAR16:
        return sampling_rate * 2
    elif audio_encoding == AudioEncoding.MULAW:
        return sampling_rate
    else:
        raise Exception("Unsupported audio encoding")


def create_conversation_id() -> str:
    return secrets.token_urlsafe(16)

def remove_non_letters_digits(text):
    return ''.join(i for i in text if i in custom_alphabet)

def print_visemes(lipsync_events):
    out = ""
    for event in lipsync_events:
        label = ALL_VISEME_LABELS.get(event["viseme_id"], event["viseme_id"])
        out += f'{event["audio_offset"]}\t{label}\n'
    return out

def save_as_wav(path, audio_data: bytes, sampling_rate: int):
    if len(audio_data) == 0:
        logger.error(f"Cannot save an empty WAV file to {path}")
        return
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        wav_file = wave.open(f, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_data)
        wav_file.close()
    logger.debug(f"Saved {len(audio_data)} audio bytes to {path}")

def trim_audio(
        sampling_rate: int,
        audio_buffer: bytearray,
        total_bytes: int,
        offset_s: float,
        duration_s: float,
    ):
    """
    Extracts / trims an audio buffer to match the given offset and duration.

    Args:
        sampling_rate: sampling rate of audio buffer
        audio_buffer: The audio buffer. May have been truncated, e.g. length may be < total_bytes
        total_bytes: The total number of bytes recorded since starting the stream
        offset_s: The offset in seconds since start of stream, to start trimming from
        duration_s: The duration in seconds to trim
    """
    bytes_per_sample = 2

    offset_bytes = int(max(0, offset_s) * bytes_per_sample * sampling_rate)
    duration_bytes = int(max(0, duration_s) * bytes_per_sample * sampling_rate)

    # Because audio_buffer only represents last x seconds (bytes may have been discarded since stream started), 
    # we need to shift the offset bytes to fit inside it
    if offset_bytes + duration_bytes > total_bytes:
            duration_bytes = total_bytes - offset_bytes
    # We may have discarded earlier audio so we need to adjust the offset
    offset_bytes -= total_bytes
    offset_bytes += len(audio_buffer)
    offset_bytes = max(0, offset_bytes)
    if duration_bytes == 0 or len(audio_buffer) <= (bytes_per_sample * sampling_rate):
        trimmed_audio = audio_buffer
    else:
        trimmed_audio = audio_buffer[offset_bytes : offset_bytes + duration_bytes]
    return trimmed_audio