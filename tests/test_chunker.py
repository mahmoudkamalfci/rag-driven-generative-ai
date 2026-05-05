# tests/test_chunker.py
import pytest
from index import load_and_chunk


def test_chunks_are_not_empty(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("A" * 1200)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    assert len(chunks) > 0


def test_chunk_size_respected(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("A" * 1200)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    for text, _ in chunks:
        assert len(text) <= 500


def test_chunk_index_sequential(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("B" * 1000)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    indices = [idx for _, idx in chunks]
    assert indices == list(range(len(chunks)))


def test_overlap_produces_shared_content(tmp_path):
    f = tmp_path / "sample.txt"
    # 600 chars: first chunk is [0:500], second starts at [450:] (500 - 50 overlap)
    f.write_text("X" * 600)
    chunks = load_and_chunk(str(f), chunk_size=500, overlap=50)
    assert len(chunks) == 2
    # The end of chunk 0 and start of chunk 1 should share 50 chars
    assert chunks[0][0][-50:] == chunks[1][0][:50]


def test_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_and_chunk(str(f), chunk_size=500, overlap=50)
