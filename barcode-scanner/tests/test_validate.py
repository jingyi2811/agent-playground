from scanner.validate import gs1_check_digit, is_valid_gtin


def test_known_ean13():
    assert is_valid_gtin("4006381333931")  # Stabilo pen, a common example
    assert not is_valid_gtin("4006381333932")


def test_known_upca():
    assert is_valid_gtin("036000291452")


def test_check_digit_computation():
    assert gs1_check_digit("400638133393") == 1


def test_rejects_non_digits_and_bad_lengths():
    assert not is_valid_gtin("LOT-1234")
    assert not is_valid_gtin("12345")


def test_normalize_upca_to_ean13():
    from scanner.validate import normalize_gtin
    assert normalize_gtin("931002516780") == "0931002516780"
    assert normalize_gtin("0931002516780") == "0931002516780"
    assert normalize_gtin("LOT-1234") == "LOT-1234"
