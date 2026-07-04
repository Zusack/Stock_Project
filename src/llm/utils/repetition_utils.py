"""
Shared detection of garbage/repetition in LLM streamed responses.
Used by both benchmark (generation) and evaluation engines to auto-stop
when the model enters a repetition loop or outputs gibberish.
"""
import re

# Gibberish: long unbroken string or oversized word in tail
GIBBERISH_CHUNK_NO_SPACE_LEN = 100
GIBBERISH_TAIL_CHARS = 300
GIBBERISH_MAX_WORD_LEN = 100

# Repetition: same phrase repeated many times in a tail window
REPETITION_WINDOW_CHARS = 4000
REPETITION_MAX_COUNT = 20
REPETITION_MIN_PHRASE_LEN = 15
REPETITION_PHRASE_LENGTHS = (20, 30, 40, 50, 60)
MIN_FULL_TEXT_LEN_FOR_REPETITION = 500


def detect_garbage_response(text_chunk: str, full_text: str) -> bool:
    """
    Returns True if the response looks like gibberish or excessive repetition,
    so the caller can cancel the stream and fail the job.

    - Gibberish: chunk with >100 chars and no spaces, or any word in last 300
      chars of full_text longer than 100 chars.
    - Repetition: in the last REPETITION_WINDOW_CHARS of full_text, the same
      phrase (length in REPETITION_PHRASE_LENGTHS) appears at least
      REPETITION_MAX_COUNT times. Only run when len(full_text) >= 500.
    """
    if not isinstance(full_text, str):
        full_text = str(full_text) if full_text else ""
    if not isinstance(text_chunk, str):
        text_chunk = str(text_chunk) if text_chunk else ""

    # --- Existing gibberish checks ---
    if len(text_chunk) > GIBBERISH_CHUNK_NO_SPACE_LEN and " " not in text_chunk:
        return True
    tail_300 = full_text[-GIBBERISH_TAIL_CHARS:]
    words = tail_300.split()
    if words:
        longest_word = max(len(w) for w in words)
        if longest_word > GIBBERISH_MAX_WORD_LEN:
            return True

    # --- Repetition check (only when enough text) ---
    if len(full_text) < MIN_FULL_TEXT_LEN_FOR_REPETITION:
        return False
    tail = full_text[-REPETITION_WINDOW_CHARS:]
    for n in REPETITION_PHRASE_LENGTHS:
        if n > len(tail):
            continue
        phrase = tail[-n:].strip()
        if len(phrase) < REPETITION_MIN_PHRASE_LEN:
            continue
        count = tail.count(phrase)
        if count >= REPETITION_MAX_COUNT:
            return True
    return False


# Thinking/reasoning block tag names used by o1, DeepSeek R1, and similar models
_THINKING_TAG_NAMES = ('think', 'reasoning', 'thought', 'thoughts')


def strip_thinking_blocks(text: str) -> str:
    """
    Remove thinking/reasoning blocks from the response so only the final answer
    is evaluated. Used for standardized benchmarks (BLEU/ROUGE/BERT) to avoid
    penalizing thinking models for outputting reasoning before the answer.

    Supports common tags: <think>, <think>, <reasoning>, <thought>, <thoughts>.
    """
    if not text or not isinstance(text, str):
        return text or ""
    result = text.strip()
    for tag_name in _THINKING_TAG_NAMES:
        # Match opening tag (with optional attributes) and everything until closing tag
        pattern = rf'<{tag_name}[^>]*>.*?</{tag_name}>'
        result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)
    if result.strip():
        return result.strip()
    return text.strip()
