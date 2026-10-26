"""Browser interaction coverage lives in scripts/verify_product_flows.py.

Keep the document's accessible control contract explicit for the Python checks.
"""
from html.parser import HTMLParser
from pathlib import Path


class Controls(HTMLParser):
    def __init__(self):super().__init__();self.ids=set();self.labels=set()
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if 'id' in values:self.ids.add(values['id'])
        if tag=='label' and 'for' in values:self.labels.add(values['for'])


def test_guidance_inputs_have_explicit_accessible_labels():
    controls=Controls();controls.feed((Path(__file__).parents[1]/'ui/index.html').read_text())
    expected={'confirmed-produce','storage-condition','storage-ripeness','storage-location','storage-spoilage','days-stored'}
    assert expected<=controls.ids and expected<=controls.labels
    assert {'produce-confidence','report-confidence','storage-result'}<=controls.ids
