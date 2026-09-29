import pytest

from tools.utils import soma_tiledb_context


def test_soma_tiledb_context_uses_configured_concurrency(monkeypatch):
    monkeypatch.setenv("SCALING_TILEDB_COMPUTE_CONCURRENCY", "2")
    monkeypatch.setenv("SCALING_TILEDB_IO_CONCURRENCY", "3")

    context = soma_tiledb_context()

    assert context.tiledb_config["sm.compute_concurrency_level"] == 2
    assert context.tiledb_config["sm.io_concurrency_level"] == 3


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_soma_tiledb_context_rejects_invalid_concurrency(monkeypatch, value):
    monkeypatch.setenv("SCALING_TILEDB_COMPUTE_CONCURRENCY", value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        soma_tiledb_context()
