def test_ask_operator_valid(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    from g3a_bench.scripts.live_bridge.live_record import ask_operator

    assert ask_operator() == 2


def test_ask_operator_retries_on_bad_input(monkeypatch):
    responses = iter(["0", "5", "abc", "3"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    from g3a_bench.scripts.live_bridge.live_record import ask_operator

    assert ask_operator() == 3
