from sentiment import preprocess as pp


def test_strip_html():
    assert pp.strip_html("Great<br /><br />film &amp; cast").split() == ["Great", "film", "&", "cast"]


def test_normalize():
    out = pp.normalize("I DIDN'T like it!! See http://x.com")
    assert "http" not in out
    assert "did not" in out
    assert out == out.lower()
    assert "!" in out.split()
