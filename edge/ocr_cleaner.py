import re
from typing import Optional

STANDARD_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BHARAT_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

CONFUSION_TO_NUM = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0"}
CONFUSION_TO_CHAR = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}

def clean_and_validate_plate(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
    if STANDARD_REGEX.match(cleaned) or BHARAT_REGEX.match(cleaned):
        return cleaned

    chars = list(cleaned)
    n = len(chars)

    # Standard HSRP positional fix: AA NN AAA NNNN or AA N AAA NNNN
    if 9 <= n <= 11:
        # Shuru ke do character letter hone chahiye (State Code)
        for idx in (0, 1):
            if chars[idx] in CONFUSION_TO_CHAR:
                chars[idx] = CONFUSION_TO_CHAR[chars[idx]]
        # Aakhri ke chaar character number hone chahiye
        for idx in (-4, -3, -2, -1):
            if chars[idx] in CONFUSION_TO_NUM:
                chars[idx] = CONFUSION_TO_NUM[chars[idx]]
        # District code digits fix (indices 2 aur 3 agar letter ban gaye hon)
        if chars[2] in CONFUSION_TO_NUM:
            chars[2] = CONFUSION_TO_NUM[chars[2]]
        if n >= 10 and chars[3].isalpha() and chars[4].isalpha() and chars[3] in CONFUSION_TO_NUM:
            # Agar format KA01AB1234 hai aur index 3 par number hona chahiye
            chars[3] = CONFUSION_TO_NUM[chars[3]]

        candidate = "".join(chars)
        if STANDARD_REGEX.match(candidate) or BHARAT_REGEX.match(candidate):
            return candidate

    return cleaned if len(cleaned) >= 6 else None