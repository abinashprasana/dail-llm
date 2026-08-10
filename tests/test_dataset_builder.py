import pytest

from dail_llm.data.dataset_builder import chunk_text, train_val_test_split


def test_chunk_text_stops_after_terminal_chunk():
    assert chunk_text("abcdefghij", chunk_chars=5, overlap=2) == [
        "abcde",
        "defgh",
        "ghij",
    ]


@pytest.mark.parametrize(
    ("chunk_chars", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_text_rejects_invalid_sizes(chunk_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_chars=chunk_chars, overlap=overlap)


def test_split_validates_ratios():
    with pytest.raises(ValueError):
        train_val_test_split("abcdef", 0.7, 0.4)


def test_split_preserves_order():
    train, validation, test = train_val_test_split("abcdefghij", 0.2, 0.2)
    assert (train, validation, test) == ("abcdef", "gh", "ij")
