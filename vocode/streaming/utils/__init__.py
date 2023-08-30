import asyncio
import audioop
import logging
import os
import secrets
from typing import Any
import wave
import struct

from ..models.audio_encoding import AudioEncoding

logger = logging.getLogger(__name__)

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

def save_as_wav(path, audio_data: bytes, sampling_rate: int):
    if len(audio_data) == 0:
        logger.error(f"Cannot save an empty WAV file to {path}")
        return
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

def hex_dump(byte_string, length=16):
    """Return a hex dump of the byte string."""
    return '\n'.join([byte_string[i:i+length].hex() for i in range(0, len(byte_string), length)])

def read_wav_file(file_path):
    """
    Reads a WAV file and returns its byte string.
    """
    with open(file_path, 'rb') as file:
        return file.read()

def has_wav_header(byte_string):
    """
    Check if a byte string starts with a WAV header.
    """
    if len(byte_string) < 12:
        return False
    return byte_string.startswith(b'RIFF') and byte_string[8:12] == b'WAVE'

def remove_wave_header(wav_byte_string):
    """
    Remove the WAV header from a byte string.
    """
    return wav_byte_string[44:]

def create_wav_dict(wav_byte_string):
    """
    Convert a bytestring to a dictionary of values hardcoded to index positions across 44B.
    
    Parameters:
    bytestring (bytearray): The bytestring to convert.
    
    Returns:
    dict: Returns values.
    """
    wav_dict = {}

    wav_dict['file_id'] = wav_byte_string[0:4]  # RIFF chunk header, big-endian
    wav_dict['file_size'] = wav_byte_string[4:8]  # Byte format for 32-bit integer, little-endian
    wav_dict['file_type'] = wav_byte_string[8:12]  # WAVE Header, requires 'fmt' and 'data', big-endian

    wav_dict['fmt_id'] = wav_byte_string[12:16]  # fmt sub-chunk header, big-endian
    wav_dict['fmt_size'] = wav_byte_string[16:20]  # Size of fmt, little-endian
    wav_dict['fmt_type'] = wav_byte_string[20:22]  # Type of format, (1 is PCM), little-endian
    wav_dict['num_channels'] = wav_byte_string[22:24]  # Number of channels, little-endian
    wav_dict['sample_rate'] = wav_byte_string[24:28]  # Sample rate in Hz, (e.g., 44100,48000), little-endian
    wav_dict['byte_rate'] = wav_byte_string[28:32]  # (Sample Rate * BitsPerSample * Channels)/8, little-endian
    wav_dict['block_align'] = wav_byte_string[32:34]  # Calculate blocks, (BitsPerSample * Channels), little-endian
    wav_dict['bits_per_sample'] = wav_byte_string[34:36]  # BitsPerSample, little-endian

    wav_dict['data_id'] = wav_byte_string[36:40]  # data sub-chunk headerl, big-endian
    wav_dict['data_size'] = wav_byte_string[40:44]  # Size of data, little-endian

    return wav_dict

def byte_to_int32(format_specifier: str, byte_string: bytearray):
    """
    Convert a bytestring to Int32 using struct.unpack
    
    Parameters:
    format_specifier (str): The format specifier for struct.unpack.
    bytestring (bytearray): The bytestring to convert.
    
    Returns:
    tuple: Returns integer.
    """
    try:
        return struct.unpack(format_specifier, byte_string)[0]

    except struct.error as e:
        print(f"An error occurred: {e}")
        return None

