# Auto-generated dialog policy config.
# You can edit these values and restart the app.

NEW_SESSION_AFTER_MIN = 360
GREETING_MAX_WORDS = 6
GREETING_MAX_CHARS = 35
GREETINGS = [
    "привет",
    "приветик",
    "здарова",
    "здравствуйте",
    "доброе утро",
    "добрый день",
    "добрый вечер",
    "hi",
    "hello",
    "hey",
    "yo",
]
GREETING_EXCLUSIONS = [
    "слово привет",
    "передай привет",
    "передайте привет",
    "приветствие",
    "в коде привет",
    "обсуждение слова привет",
    "перевод привет",
]
DEFAULT_WARMTH_LEVEL = 0.58
DEFAULT_SARCASM_LEVEL = 0.24
DEFAULT_STRICTNESS_LEVEL = 0.56
DEFAULT_VERBOSITY_LEVEL = 0.46
TECHNICAL_MAX_SARCASM = 0.08

TERMS_ENABLED = True
TERMS_LIST = ["милашка"]
TERMS_COOLDOWN_TURNS = 6
TERMS_COOLDOWN_SECONDS = 900
TERMS_MAX_PER_SESSION = 3
TERMS_INSERT_PROBABILITY = 0.35
TERMS_BAN_SCOPE = "session"
TERMS_DISABLE_PATTERNS = [
    "не называй",
    "прекрати называть",
    "не зови",
    "прекрати звать",
    "stop calling me",
    "don't call me",
    "no pet names",
]
TERMS_ENABLE_PATTERNS = [
    "можно снова",
    "можешь снова",
    "можно называть",
    "call me again",
    "you can call me",
]
