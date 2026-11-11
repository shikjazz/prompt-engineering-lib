"""Tests for the output parsers."""

from prompt_lib.parser import JsonParser, ListParser, RegexParser, KeyValueParser


def test_json_parse_plain():
    parser = JsonParser()
    result = parser.parse('{"key": "value", "num": 42}')
    assert result.success
    assert result.data == {"key": "value", "num": 42}


def test_json_parse_with_fences():
    parser = JsonParser()
    text = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
    result = parser.parse(text)
    assert result.success
    assert result.data == {"a": 1}


def test_json_parse_with_schema_validation():
    parser = JsonParser(schema={"required": ["name", "age"]})
    result = parser.parse('{"name": "Alice", "age": 30}')
    assert result.success

    result2 = parser.parse('{"name": "Bob"}')
    assert not result2.success
    assert "Missing required key" in result2.error


def test_json_parse_failure():
    parser = JsonParser()
    result = parser.parse("not json at all")
    assert not result.success


def test_list_parser():
    parser = ListParser()
    text = "1. First item\n2. Second item\n3. Third item"
    result = parser.parse(text)
    assert result.success
    assert result.data == ["First item", "Second item", "Third item"]


def test_list_parser_bullets():
    parser = ListParser()
    text = "- Apple\n- Banana\n- Cherry"
    result = parser.parse(text)
    assert result.data == ["Apple", "Banana", "Cherry"]


def test_list_parser_min_items():
    parser = ListParser(min_items=5)
    result = parser.parse("1. One\n2. Two")
    assert not result.success
    assert "at least 5" in result.error


def test_regex_parser_named_groups():
    parser = RegexParser(r"Score: (?P<score>\d+)/(?P<total>\d+)")
    result = parser.parse("The result is Score: 85/100 overall.")
    assert result.success
    assert result.data == {"score": "85", "total": "100"}


def test_regex_parser_no_match():
    parser = RegexParser(r"Score: (\d+)")
    result = parser.parse("No score here")
    assert not result.success


def test_kv_parser():
    parser = KeyValueParser()
    text = "Name: Alice\nAge: 30\nCity: Berlin"
    result = parser.parse(text)
    assert result.success
    assert result.data == {"Name": "Alice", "Age": "30", "City": "Berlin"}


def test_kv_parser_required_keys():
    parser = KeyValueParser(expected_keys=["Name", "Age"])
    result = parser.parse("Name: Alice")
    assert not result.success
    assert "Missing keys" in result.error
