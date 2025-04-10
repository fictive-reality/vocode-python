# Increased chunk size to better work with current lipsync implementation
TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS = 0.5
PER_CHUNK_ALLOWANCE_SECONDS = 0.05
ALLOWED_IDLE_TIME = 15
SENTENCE_ENDINGS = [".", "!", "?", "\n"]
CHECK_HUMAN_PRESENT_MESSAGE_CHOICES = [
    "Hello?",
    "Are you there?",
    "Are you still there?",
    "Hi, are you there?",
]
