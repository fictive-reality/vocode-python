from vocode.streaming.models.audio_encoding import AudioEncoding
from typing import Optional
from opentelemetry.trace import Span

class BaseOutputDevice:
    def __init__(self, sampling_rate: int, audio_encoding: AudioEncoding):
        self.sampling_rate = sampling_rate
        self.audio_encoding = audio_encoding

    def start(self):
        pass

    def consume_nonblocking(self, chunk: bytes, lipsync_events: Optional[list] = None, span: Optional[Span] = None):
        raise NotImplemented
    
    def maybe_send_mark_nonblocking(self, message):
        pass

    def terminate(self):
        pass
