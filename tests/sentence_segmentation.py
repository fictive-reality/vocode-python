import re


def sentence_boundary_detection(text):
    combined_pattern = r"(\b(?:[a-zA-Z]\.){2,}|\d*\.?\d+)"
    replacements = []

    def replacer(match):
        value = match.group(0)
        replacements.append(value)
        return f"PLACEHOLDER_{len(replacements)-1}"

    text_with_placeholders = re.sub(combined_pattern, replacer, text)
    sentences = re.split(r"(?<=[.!?])\s+", text_with_placeholders)

    for i, replacement in enumerate(replacements):
        sentences = [
            sentence.replace(f"PLACEHOLDER_{i}", replacement) for sentence in sentences
        ]

    return sentences
