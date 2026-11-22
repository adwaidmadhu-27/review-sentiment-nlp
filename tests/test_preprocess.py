from sentiment import preprocess as pp


def test_strip_html():
    assert pp.strip_html("Great<br /><br />film &amp; cast").split() == ["Great", "film", "&", "cast"]
