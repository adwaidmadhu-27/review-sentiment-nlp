from sentiment import preprocess as pp


def test_strip_html():
    assert pp.strip_html("Great<br /><br />film &amp; cast").split() == ["Great", "film", "&", "cast"]


def test_normalize():
    out = pp.normalize("I DIDN'T like it!! See http://x.com")
    assert "http" not in out
    assert "did not" in out
    assert out == out.lower()
    assert "!" in out.split()


def test_mark_negation_window_and_clause_end():
    out = pp.mark_negation("this is not good at all . but fun")
    assert out == "this is not NOT_good NOT_at NOT_all . but fun"


def test_negation_stops_at_but():
    assert pp.mark_negation("not funny but clever") == "not NOT_funny but clever"


def test_clean_combines_steps():
    assert pp.clean("It wasn't good<br />") == "it was not NOT_good"
    assert pp.clean("It wasn't good", negation=False) == "it was not good"
