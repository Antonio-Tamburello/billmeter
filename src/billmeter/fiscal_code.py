"""Italian codice fiscale: validation and generation (public algorithm).

Generation is used only to create *fake but valid* tax codes for the synthetic dataset.
"""

from __future__ import annotations

import re
from datetime import date

_ODD = {
    **dict(zip("0123456789", [1, 0, 5, 7, 9, 13, 15, 17, 19, 21], strict=True)),
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20,
                                              11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23], strict=True)),
}
_EVEN = {**{d: int(d) for d in "0123456789"}, **{c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}}
_MONTHS = "ABCDEHLMPRST"
# Digits can be replaced by letters (omocodia) when two people would get the same code.
_DIGIT = "[0-9LMNPQRSTUV]"
_PATTERN = re.compile(rf"^[A-Z]{{6}}{_DIGIT}{{2}}[A-Z]{_DIGIT}{{2}}[A-Z]{_DIGIT}{{3}}[A-Z]$")
_VOWELS = "AEIOU"


def check_char(first15: str) -> str:
    """Control character computed on the first 15 characters."""
    total = sum(_ODD[c] if i % 2 == 0 else _EVEN[c] for i, c in enumerate(first15))
    return chr(ord("A") + total % 26)


def is_valid(code: str) -> bool:
    """True if the string is a well-formed person tax code with the right control char."""
    code = code.strip().upper()
    return bool(_PATTERN.match(code)) and check_char(code[:15]) == code[15]


def _letters(text: str) -> tuple[str, str]:
    text = re.sub(r"[^A-Z]", "", text.upper())
    consonants = "".join(c for c in text if c not in _VOWELS)
    vowels = "".join(c for c in text if c in _VOWELS)
    return consonants, vowels


def _surname_part(surname: str) -> str:
    consonants, vowels = _letters(surname)
    return (consonants + vowels + "XXX")[:3]


def _name_part(name: str) -> str:
    consonants, vowels = _letters(name)
    if len(consonants) >= 4:
        return consonants[0] + consonants[2] + consonants[3]
    return (consonants + vowels + "XXX")[:3]


def generate(first_name: str, last_name: str, birth: date, female: bool, municipality_code: str) -> str:
    """Build a valid codice fiscale (municipality_code is the 4-char Belfiore code, e.g. 'H501')."""
    day = birth.day + (40 if female else 0)
    first15 = (
        _surname_part(last_name)
        + _name_part(first_name)
        + f"{birth.year % 100:02d}"
        + _MONTHS[birth.month - 1]
        + f"{day:02d}"
        + municipality_code.upper()
    )
    return first15 + check_char(first15)
