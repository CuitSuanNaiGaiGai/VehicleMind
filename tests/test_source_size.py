from scripts.check_source_size import find_oversized_sources


def test_source_files_respect_size_policy() -> None:
    assert find_oversized_sources() == []
