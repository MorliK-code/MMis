import re
from typing import Dict

TEASE_MARKERS = [
    "ну ты", "ага", "конечно", "смешной", "красава", "умник", "лол", "ахах", "😉", "😄"
]

def infer_assistant_traits(assistant_text: str) -> Dict:
    text = assistant_text.lower()

    patch = {}
    emoji_count = len(re.findall(r"[😀-🙏😉-🫶❤️😂🤣😄😅😊😍🤪😈]", assistant_text))
    if emoji_count >= 2:
        patch["emoji_level"] = 2
    elif emoji_count == 1:
        patch["emoji_level"] = 1
    
    if any(m in text for m in TEASE_MARKERS):
        patch["style"] = "легкая ирония"
        patch["humor_level"] = 2

    return patch