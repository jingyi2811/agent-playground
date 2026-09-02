"""Checksum validation for common retail formats.

Decoders already verify the check digit, but doing it explicitly shows the
GS1 rule and lets us reject values that decode cleanly but are not retail codes.
"""


def gs1_check_digit(digits: str) -> int:
    """Check digit for EAN-8, EAN-13, UPC-A, GTIN-14 (all use the same rule)."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        weight = 3 if i % 2 == 0 else 1
        total += int(ch) * weight
    return (10 - total % 10) % 10


def is_valid_gtin(code: str) -> bool:
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    return gs1_check_digit(code[:-1]) == int(code[-1])


def normalize_gtin(code: str) -> str:
    """Pad numeric codes to 13 digits so UPC-A and EAN-13 compare equal.

    UPC-A is EAN-13 with a leading zero; decoders report whichever they
    detected, so comparisons must normalize first.
    """
    if code.isdigit() and len(code) in (8, 12):
        return code.zfill(13)
    return code
