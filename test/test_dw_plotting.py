from plotting.geneformer_dw import _merge_sequence_segments


def test_merge_sequence_segments_combines_resume_and_prefers_longer_overlap() -> None:
    initial = [(0, 10.0, None), (1, 9.0, None), (2, 8.0, None)]
    resumed = [(2, 80.0, None), (3, 7.0, None)]

    merged = _merge_sequence_segments(
        [
            ("initial", initial),
            ("resumed", resumed),
        ]
    )

    assert merged == [
        (0, 10.0, None),
        (1, 9.0, None),
        (2, 8.0, None),
        (3, 7.0, None),
    ]
