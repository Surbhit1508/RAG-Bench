from ragbench.chunking import chunk_fixed, chunk_sentence, chunk_document


def test_chunk_fixed_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_fixed(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    # overlap: last few words of chunk N should reappear at start of chunk N+1
    tail = chunks[0].split()[-5:]
    head = chunks[1].split()[:5]
    assert tail == head


def test_chunk_fixed_empty_text():
    assert chunk_fixed("", chunk_size=10, chunk_overlap=2) == []


def test_chunk_sentence_keeps_sentences_whole():
    text = "First sentence here. Second sentence here. Third one too."
    chunks = chunk_sentence(text, chunk_size=6, chunk_overlap=2)
    for chunk in chunks:
        assert chunk.strip().endswith(".")


def test_chunk_document_dispatches_correctly():
    text = "One sentence. Another sentence."
    assert chunk_document(text, "fixed", 10, 2) == chunk_fixed(text, 10, 2)
    assert chunk_document(text, "sentence", 10, 2) == chunk_sentence(text, 10, 2)


def test_chunk_document_rejects_unknown_strategy():
    try:
        chunk_document("text", "bogus", 10, 2)
        assert False, "should have raised"
    except ValueError:
        pass
