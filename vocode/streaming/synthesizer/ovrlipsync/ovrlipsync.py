import asyncio
import subprocess
import pathlib
import sys
import json

# If we are in our own repo, we locate the ProcessWAV.exe in the Lib/Win64 folder
# Otherwise we assume it's been added into the same folder
exec_path = pathlib.Path(__file__).parent.parent.joinpath("Lib/Win64").absolute()
if not exec_path.exists():
    exec_path = pathlib.Path(__file__).parent.absolute()

process_exe = exec_path.joinpath('ProcessWAV.exe')

class OVRLipsyncProcessor:
    def __init__(self, sample_rate: int, buffer_ms: int = 10, timeout: float = 0.5):
        self.sample_rate = sample_rate
        self.buffer_ms = buffer_ms
        # ProcessWAV calculates a buffer_size, but counts 16-bit samples
        # So our buffer in bytes should be 2x the size
        self.buffer_size = 2 * int(sample_rate * buffer_ms / 1000)
        self.process = None
        # Timeout in seconds to wait for processing of ONE frame
        self.timeout = timeout
        self.lock = asyncio.Lock()

    def start(self):
        self.process = subprocess.Popen(
            ["wine", process_exe, str(self.sample_rate), str(self.buffer_ms)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.terminate()
            self.process = None
    
    async def process_frame(self, frame_data: bytes):
        if len(frame_data) != self.buffer_size:
            raise ValueError(f"Frame data must be {self.buffer_size} bytes")
        async with self.lock:
            self.process.stdin.write(frame_data)
            self.process.stdin.flush()
            try:
                stdout_line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, self.process.stdout.readline),
                    self.timeout
                )
                return stdout_line.strip()
            except asyncio.CancelledError:
                return None
            except asyncio.TimeoutError:
                raise TimeoutError(f"Processing frame timed out after {self.timeout} seconds")

    async def detect_lipsync(self, audio_data: bytes, audio_offset: float = 0.0):
        """Loops through audio_data in buffer_size chunks and processes each frame, returning a lipsync event list.
        Audio data should be raw PCM data, 16-bit samples, same sample rate as the OVRLipsyncProcessor has been defined for."""

        byte_offset = 0
        lipsync_events = []
        last_viseme = None
        if audio_data[0:4] == b"RIFF":
            # Skip the WAV header to not get buffer counts off
            byte_offset = 44
        while byte_offset < len(audio_data):
            chunk = audio_data[byte_offset:byte_offset + self.buffer_size]
            byte_offset += self.buffer_size
            if len(chunk) == self.buffer_size:
                try:
                    viseme = await self.process_frame(chunk)
                    if viseme and viseme != last_viseme:
                        decoded = viseme.decode()
                        # Quick hack to fix wrong labels sent by ProcessWAV.exe
                        if decoded == "ih":
                            decoded = "I"
                        elif decoded == "oh":
                            decoded = "O"
                        elif decoded == "ou":
                            decoded = "U"
                        lipsync_events.append({"audio_offset": audio_offset, "viseme_id": f"ovr_{decoded}"})
                    last_viseme = viseme
                except TimeoutError:
                    pass
                audio_offset = round(audio_offset + self.buffer_ms / 1000, 2)
        return lipsync_events
            

async def main(filepath):
    processor = OVRLipsyncProcessor(24000)
    processor.start()

    with open(filepath, 'rb') as wav:
        audio_data = wav.read()

    lipsync_events = await processor.detect_lipsync(audio_data[44:]) # Skip the WAV header as it throws buffer counts off
    output = json.dumps(lipsync_events)
    print(output)
    pathlib.Path("visemes.json").write_text(output)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ovrlipsync.py <filepath.wav>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))