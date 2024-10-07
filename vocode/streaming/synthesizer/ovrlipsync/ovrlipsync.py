import asyncio
import logging
import subprocess
import pathlib
import sys
import json
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# If we are in our own repo, we locate the ProcessWAV.exe in the Lib/Win64 folder
# Otherwise we assume it's been added into the same folder
exec_path = pathlib.Path(__file__).parent.parent.joinpath("Lib/Win64").absolute()
if not exec_path.exists():
    exec_path = pathlib.Path(__file__).parent.absolute()

process_exe = exec_path.joinpath('ProcessWAV.exe')

ovr_viseme_ids = [
  "ovr_sil", "ovr_PP", "ovr_FF", "ovr_TH", "ovr_DD",
  "ovr_kk", "ovr_CH", "ovr_SS", "ovr_nn", "ovr_RR",
  "ovr_aa", "ovr_E", "ovr_I", "ovr_O", "ovr_U",
]

class OVRLipsyncProcessor:
    def __init__(self, sample_rate: int, buffer_ms: int = 10, timeout: float = 5, print_as_array: bool = False):
        self.sample_rate = sample_rate
        self.buffer_ms = buffer_ms
        # ProcessWAV calculates a buffer_size, but counts 16-bit samples
        # So our buffer in bytes should be 2x the size
        self.buffer_size = 2 * int(sample_rate * buffer_ms / 1000)
        self.process = None
        # Timeout in seconds to wait for processing of ONE frame
        self.timeout = timeout
        self.lock = asyncio.Lock()
        self.command = ["wine", process_exe, str(self.sample_rate), str(self.buffer_ms)]
        self.print_as_array = print_as_array
        if print_as_array:
            self.command.append("--print-as-array")

    def parse_array(self, array_string: str):
        string_elements = array_string.split(';')
        float_array = [float(element.strip()) for element in string_elements]
        return float_array
    
    def analyze_viseme_arrays(self, viseme_arrays, window_size=3):
        stream = np.array(viseme_arrays)
        
        def moving_avg(array, window_size):
            return np.convolve(array, np.ones(window_size) / window_size, mode='valid')
        
        # Apply moving average on each signal
        smoothed_stream = np.array([moving_avg(signal, window_size) for signal in stream.T]).T

        # Find the index of the dominant signal at each timestamp
        dominant_signals = np.argmax(smoothed_stream, axis=1)

        # Iterate over dominant signals to detect changes
        last_signal = None
        lipsync_events = []
        for i, signal in enumerate(dominant_signals):
            if signal != last_signal:
                audio_offset = i * self.buffer_ms / 1000.0  # Calculate audio offset in seconds
                viseme_id = ovr_viseme_ids[signal]  # Get viseme ID from the table
                lipsync_events.append({"audio_offset": audio_offset, "viseme_id": viseme_id})
                last_signal = signal

        return lipsync_events

    async def start(self):
        logger.info(f"Starting the process {self.command}")
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        asyncio.create_task(self.read_stderr())

    async def close(self):
        if self.process is not None:
            self.process.stdin.close()
            await self.process.wait()  # Wait for the subprocess to terminate
            self.process = None
    
    async def read_stderr(self):
        while self.process:
            line = await self.process.stderr.readline()
            if line:
                logger.info(f"ProcessWAV.exe: {line.decode().strip()}")
            else:
                break
    
    async def process_frame(self, frame_data: bytes):
        async with self.lock:
            try:
                self.process.stdin.write(frame_data)
                await self.process.stdin.drain()
                stdout_line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    self.timeout
                )
                return stdout_line.strip()
            except asyncio.CancelledError:
                return None # We were cancelled, return None
            except Exception:
                logger.exception("Error processing frame, restarting ProcessWAV process")
                await self.close()
                await self.start()
                return await self.process_frame(frame_data)

    async def detect_lipsync(self, audio_data: bytes, audio_offset: float = 0.0):
        """Loops through audio_data in buffer_size chunks and processes each frame, returning a lipsync event list.
        Audio data should be raw PCM data, 16-bit samples, same sample rate as the OVRLipsyncProcessor has been defined for."""

        byte_offset = 0
        lipsync_events = []
        last_viseme = None
        viseme_arrays = []
        if audio_data[0:4] == b"RIFF":
            # Skip the WAV header to not get buffer counts off
            byte_offset = 44
        logger.info(f"Processing {len(audio_data)} bytes of audio data")
        while byte_offset < len(audio_data):
            chunk = audio_data[byte_offset:byte_offset + self.buffer_size]
            byte_offset += self.buffer_size
            if len(chunk) == self.buffer_size:
                viseme = (await self.process_frame(chunk)).decode()
                if self.print_as_array:
                    viseme_arrays.append(self.parse_array(viseme))
                else:
                    if viseme and viseme != last_viseme:
                        lipsync_events.append({"audio_offset": audio_offset, "viseme_id": f"ovr_{viseme}"})
                    last_viseme = viseme
                audio_offset = round(audio_offset + self.buffer_ms / 1000, 2)
        if self.print_as_array:        
            return self.analyze_viseme_arrays(viseme_arrays)
        else:
            return lipsync_events
            

async def main(filepath):
    processor = OVRLipsyncProcessor(24000)
    await processor.start()

    with open(filepath, 'rb') as wav:
        audio_data = wav.read()

    lipsync_events = await processor.detect_lipsync(audio_data[44:]) # Skip the WAV header as it throws buffer counts off
    output = json.dumps(lipsync_events)
    print(output)
    pathlib.Path("visemes.json").write_text(output)
    await processor.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ovrlipsync.py <filepath.wav>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))