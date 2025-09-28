import json
from typing import Iterable, List

import pytest

import data
import main
import stats
import summary


@pytest.fixture
def dataset_factory(tmp_path, monkeypatch):
    def _create(lines: Iterable[str]):
        dataset_file = tmp_path / "dataset.tff"
        dataset_file.write_text("\n".join(lines))
        monkeypatch.setattr(data, "DATA_SOURCE", str(dataset_file))
        return dataset_file

    return _create


@pytest.fixture
def patch_stats_dataset(monkeypatch):
    def _apply(entries: List[dict]):
        calls: List[None] = []

        def fake_read_dataset():
            calls.append(None)
            return entries

        monkeypatch.setattr(stats, "read_dataset", fake_read_dataset)
        return calls

    return _apply


@pytest.mark.parametrize(
    "text,dataset_entries,expected_vowels,expected_consonants,expected_words,expected_mood",
    [
        (
            "Hello, World...",
            [
                {"word": "hello", "polarity": "positive"},
                {"word": "world", "polarity": "positive"},
            ],
            {"e": 1, "o": 2},
            {"h": 1, "l": 3, "w": 1, "r": 1, "d": 1},
            {"hello": 1, "world": 1},
            "positive",
        ),
        (
            "Foo bar???",
            [
                {"word": "foo", "polarity": "positive"},
                {"word": "bar", "polarity": "negative"},
            ],
            {"o": 2, "a": 1},
            {"f": 1, "b": 1, "r": 1},
            {"foo": 1, "bar": 1},
            "neutral",
        ),
        (
            "Loren ipsum",
            [
                {"word": "loren", "polarity": "negative"},
                {"word": "ipsum", "polarity": "negative"},
            ],
            {"o": 1, "e": 1, "i": 1, "u": 1},
            {"l": 1, "r": 1, "n": 1, "p": 1, "s": 1, "m": 1},
            {"loren": 1, "ipsum": 1},
            "negative",
        ),
        (
            "",
            [],
            {},
            {},
            {},
            "neutral",
        ),
    ],
    ids=["positive", "neutral", "negative", "empty"],
)
def test_report_summary_variants(
    patch_stats_dataset,
    text,
    dataset_entries,
    expected_vowels,
    expected_consonants,
    expected_words,
    expected_mood,
):
    # Arrange
    calls = patch_stats_dataset(dataset_entries)

    # Act
    vowel_counts, consonant_counts, word_counts, mood = stats.report_summary(text)

    # Assert
    assert vowel_counts == expected_vowels
    assert consonant_counts == expected_consonants
    assert word_counts == expected_words
    assert mood == expected_mood
    assert len(calls) == 1


def test_get_mood_returns_neutral_when_no_matches(patch_stats_dataset):
    # Arrange
    patch_stats_dataset([
        {"word": "joy", "polarity": "positive"},
        {"word": "sorrow", "polarity": "negative"},
    ])

    # Act
    mood = stats.get_mood("unknown tokens only")

    # Assert
    assert mood == "neutral"


def test_num_words_normalizes_tokens():
    # Arrange
    text = "Hello, HELLO! friend? friend..."

    # Act
    counts = stats.num_words(text)

    # Assert
    assert counts == {"hello": 2, "friend": 2}


def test_num_vowels_counts_only_letters():
    # Arrange
    text = "Aerial?! 123"

    # Act
    counts = stats.num_vowels(text)

    # Assert
    assert counts == {"a": 2, "e": 1, "i": 1}


def test_read_dataset_success(dataset_factory):
    # Arrange
    dataset_factory(
        [
            "type=weaksubj len=1 word1=wow pos1=adj stemmed1=y priorpolarity=positive",
            "type=strongsubj len=2 word1=great pos1=verb stemmed1=n priorpolarity=negative",
        ]
    )

    # Act
    entries = data.read_dataset()

    # Assert
    assert entries == [
        {
            "type": "weaksubj",
            "len": "1",
            "word": "wow",
            "pos": "adj",
            "stemmed": "y",
            "polarity": "positive",
        },
        {
            "type": "strongsubj",
            "len": "2",
            "word": "great",
            "pos": "verb",
            "stemmed": "n",
            "polarity": "negative",
        },
    ]


def test_read_dataset_missing_file(tmp_path, monkeypatch):
    # Arrange
    missing_file = tmp_path / "missing.tff"
    monkeypatch.setattr(data, "DATA_SOURCE", str(missing_file))

    # Act / Assert
    with pytest.raises(RuntimeError):
        data.read_dataset()


def test_print_summary_from_file_missing(monkeypatch, capsys):
    # Arrange
    missing_name = "no_such_file.txt"
    monkeypatch.setattr(summary, "os", __import__("os"))
    monkeypatch.setattr(summary.os.path, "exists", lambda path: False)

    # Act
    summary.print_summary_from_file(missing_name, verbosity=0)
    captured = capsys.readouterr()

    # Assert
    assert f"Cannot find file : {missing_name}" in captured.out


def test_print_summary_from_string_json(monkeypatch, capsys):
    # Arrange
    text = "aba"
    expected_payload = {
        "vowel_count": {"a": 2},
        "consonant_count": {"b": 1},
        "word_count": {"aba": 1},
        "mood": "neutral",
        "content": text,
    }

    def fake_report_json_summary(content: str):
        assert content == text
        return expected_payload

    monkeypatch.setattr(summary, "report_json_summary", fake_report_json_summary)

    # Act
    summary.print_summary_from_string(text, verbosity=1, json_mode=True)
    captured = capsys.readouterr()

    # Assert
    payload = json.loads(captured.out)
    assert payload == {**expected_payload, "name": text}


def test_print_summary_from_url_json_mode(monkeypatch, capsys):
    # Arrange
    class FakeResponse:
        content = "hello world"

    get_calls = []
    report_calls = []

    def fake_get(url):
        get_calls.append(url)
        return FakeResponse()

    def fake_report_json(content):
        report_calls.append(content)
        return {"content": content, "mood": "neutral", "word_count": {}}

    monkeypatch.setattr(summary.requests, "get", fake_get)
    monkeypatch.setattr(summary, "report_json_summary", fake_report_json)

    # Act
    summary.print_summary_from_url("http://fake-url.test", verbosity=0, json_mode=True)
    captured = capsys.readouterr()

    # Assert
    payload = json.loads(captured.out)
    assert payload == {
        "name": "http://fake-url.test",
        "content": "hello world",
        "mood": "neutral",
        "word_count": {},
    }
    assert get_calls == ["http://fake-url.test"]
    assert report_calls == ["hello world"]


def test_print_summary_from_url_connection_error(monkeypatch, capsys):
    # Arrange
    get_calls = []

    def fake_get(url):
        get_calls.append(url)
        raise summary.requests.exceptions.ConnectionError

    monkeypatch.setattr(summary.requests, "get", fake_get)

    # Act
    summary.print_summary_from_url("http://fake-url-error.test", verbosity=0, json_mode=False)
    captured = capsys.readouterr()

    # Assert
    assert "Invalid url: http://fake-url-error.test" in captured.out
    assert get_calls == ["http://fake-url-error.test"]


def test_main_dispatch(monkeypatch):
    # Arrange
    calls = {"string": [], "file": [], "url": []}
    glob_calls = []

    def fake_summary_from_string(text, verbosity, json_mode):
        calls["string"].append((text, verbosity, json_mode))

    def fake_summary_from_file(filename, verbosity, json_mode):
        calls["file"].append((filename, verbosity, json_mode))

    def fake_summary_from_url(url, verbosity, json_mode):
        calls["url"].append((url, verbosity, json_mode))

    def fake_glob(pattern):
        glob_calls.append(pattern)
        return ["expanded.txt"]

    monkeypatch.setattr(main, "print_summary_from_string", fake_summary_from_string)
    monkeypatch.setattr(main, "print_summary_from_file", fake_summary_from_file)
    monkeypatch.setattr(main, "print_summary_from_url", fake_summary_from_url)
    monkeypatch.setattr(main.glob, "glob", fake_glob)

    namespace = main.argparse.Namespace(
        s=["Hello"],
        f=["*.txt"],
        u=["http://fake-url.test"],
        v=2,
        json=True,
    )
    monkeypatch.setattr(main.argparse.ArgumentParser, "parse_args", lambda self: namespace)

    # Act
    main.main()

    # Assert
    assert calls["string"] == [("Hello", 2, True)]
    assert calls["file"] == [("expanded.txt", 2, True)]
    assert calls["url"] == [("http://fake-url.test", 2, True)]
    assert glob_calls == ["*.txt"]
