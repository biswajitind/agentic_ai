from pathlib import Path

import pytest
import yaml

from sources import load_config


def write_config(tmp_path: Path, **overrides) -> Path:
    data = {
        "source": {"location": "./docs", "recursive": False},
        "vector_store": {"directory": "./vectors", "collection": "test"},
        "chunking": {"size": 500, "overlap": 50},
        "retrieval": {},
        "models": {
            "provider": "openai",
            "chat": "chat-model",
            "embedding": "embedding-model",
        },
    }
    for section, values in overrides.items():
        data[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_config_applies_defaults_and_resolves_paths(tmp_path):
    path = write_config(tmp_path)

    config = load_config(path)

    assert config.source.max_depth == 2
    assert config.retrieval.top_k == 4
    assert config.source.location == str(tmp_path / "docs")
    assert config.vector_store.directory == tmp_path / "vectors"


def test_load_config_preserves_provider_for_model_factory(tmp_path):
    path = write_config(tmp_path, models={"provider": "anthropic"})

    config = load_config(path)

    assert config.models.provider == "anthropic"


@pytest.mark.parametrize("missing", ["source", "vector_store", "chunking", "models"])
def test_load_config_rejects_missing_required_section(tmp_path, missing):
    path = write_config(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data[missing]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match=missing):
        load_config(path)


def test_load_config_rejects_overlap_not_smaller_than_size(tmp_path):
    path = write_config(tmp_path, chunking={"size": 100, "overlap": 100})

    with pytest.raises(ValueError, match="overlap"):
        load_config(path)


def test_load_config_rejects_negative_depth(tmp_path):
    path = write_config(tmp_path, source={"max_depth": -1})

    with pytest.raises(ValueError, match="max_depth"):
        load_config(path)


def test_load_config_rejects_non_boolean_recursive_value(tmp_path):
    path = write_config(tmp_path, source={"recursive": "false"})

    with pytest.raises(ValueError, match="source.recursive"):
        load_config(path)


def test_repository_sample_config_is_valid():
    config = load_config(Path(__file__).parents[1] / "config.yaml")

    assert config.vector_store.collection == "pasha"
    assert config.source.recursive is False
