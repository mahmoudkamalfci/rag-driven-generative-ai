from data_vector_store import chunk_text

def test_chunk_text():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[2]) == 500
