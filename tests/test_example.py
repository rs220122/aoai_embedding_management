from src import example


def test_example_success():
    result = example.concat_2string("aaa", "bbb")
    assert result == "aaabbb"
