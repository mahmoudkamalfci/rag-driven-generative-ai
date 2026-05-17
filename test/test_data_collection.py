from data_collection_preparation import clean_text

def test_clean_text():
    raw = "Space exploration is the use of astronomy [1] and space technology [2]."
    expected = "Space exploration is the use of astronomy  and space technology ."
    assert clean_text(raw) == expected
