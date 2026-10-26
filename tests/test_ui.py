from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_ui_has_no_shelf_life_feature():
    html = (ROOT / "ui" / "index.html").read_text().lower()
    javascript = (ROOT / "ui" / "app.js").read_text().lower()
    assert "use-period" not in html
    assert "shelf-card" not in html
    assert "shelf-life" not in html
    assert "shelf-life" not in javascript


def test_uncertain_results_use_produce_score_and_neutral_copy():
    javascript = (ROOT / "ui" / "app.js").read_text()
    assert "uncertain ? result.produceConfidence : result.freshnessConfidence" in javascript
    assert "Top produce score" in javascript
    assert "Top produce candidate" in javascript
