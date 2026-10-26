"""Check kit assets, navigation, saved scan states and responsive layouts.

Saved history fixtures exist only in this isolated browser context. Real model
and upload checks live in verify_product_flows.py and verify_scan_sources.py.
"""
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

out = Path('runs/kit-ui-verification')
out.mkdir(parents=True, exist_ok=True)
checks = []
errors = []
failed_assets = []

with sync_playwright() as tool:
    browser = tool.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 390, 'height': 844}, reduced_motion='reduce')
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('response', lambda response: failed_assets.append(response.url)
            if '/assets/' in response.url and response.status >= 400 else None)
    page.goto('http://localhost:8000')
    expect(page.locator('.home-history')).to_contain_text('No scans yet')
    assert page.evaluate('getComputedStyle(document.documentElement).getPropertyValue("--orange").trim()') == '#FF7417'
    assert page.evaluate('getComputedStyle(document.body).fontFamily').startswith('Inter,')
    # Give newly visible masks and the browser compositor time before screenshots.
    page.wait_for_timeout(300)
    page.screenshot(path=str(out / 'home-empty.png'), full_page=True)
    note = page.locator('#home-screen > .small-note').bounding_box()
    nav = page.locator('.bottom-nav').bounding_box()
    assert note['y'] + note['height'] <= nav['y'], (note, nav)
    checks.append('Kit colors, font fallback, vector artwork and labeled navigation render; Home note fits above navigation at 390x844')

    fixtures = [
        {'produce': 'cucumber', 'confirmedProduce': 'banana', 'status': 'ok', 'confidence': 1,
         'produceConfidence': 1, 'capturedAt': '2026-10-26T08:00:00Z'},
        {'produce': 'tomato', 'confirmedProduce': 'tomato', 'status': 'ok', 'confidence': .95,
         'produceConfidence': .95, 'capturedAt': '2026-10-26T07:00:00Z'},
        {'produce': 'orange', 'status': 'uncertain', 'confidence': .3,
         'produceConfidence': .3, 'capturedAt': '2026-10-26T06:00:00Z'},
    ]
    page.evaluate('(items) => { localStorage.setItem("freshpick-history", JSON.stringify(items)); renderHistory(); }', fixtures)
    expect(page.locator('.home-history')).to_contain_text('Banana · Label confirmed')
    expect(page.locator('.home-history')).to_contain_text('model suggested Cucumber')
    expect(page.locator('.home-history')).not_to_contain_text('100%')
    page.wait_for_timeout(100)
    page.screenshot(path=str(out / 'home-corrected-fixture.png'), full_page=True)
    page.locator('.home-history').click()
    expect(page.locator('#history-screen')).to_be_visible()
    expect(page.locator('#saved-scan-count')).to_have_text('3')
    expect(page.locator('#confirmed-scan-count')).to_have_text('2')
    expect(page.locator('.history-item')).to_have_count(3)
    expect(page.get_by_role('button', name='Insights', exact=True)).to_have_attribute('aria-current', 'page')
    page.wait_for_timeout(200)
    page.screenshot(path=str(out / 'insights-fixture.png'), full_page=True)
    checks.append('Recent-scan card opens Insights; fixture summaries count saved scans and confirmed labels; corrections keep model identity separate')

    page.get_by_role('button', name='Home', exact=True).click()
    for item, expected in [(fixtures[1], 'Model confidence: high · uncalibrated'),
                           (fixtures[2], 'Uncertain identification')]:
        page.evaluate('(item) => { localStorage.setItem("freshpick-history", JSON.stringify([item])); renderHistory(); }', item)
        expect(page.locator('.home-history')).to_contain_text(expected)
    checks.append('Recent-scan copy distinguishes uncertain predictions and uncalibrated confidence')

    page.evaluate('''() => { navigator.mediaDevices.getUserMedia = async () => {
        throw new DOMException('Simulated missing camera', 'NotFoundError');
    }; }''')
    page.locator('#home-screen [data-action="scan"]').click()
    expect(page.locator('#scan-notice')).to_contain_text('No camera was found')
    expect(page.locator('[data-action="flash"]')).to_be_disabled()
    expect(page.locator('[data-action="capture"]')).to_be_disabled()
    expect(page.locator('.bottom-nav')).not_to_be_visible()
    page.wait_for_timeout(200)
    page.screenshot(path=str(out / 'camera-unavailable.png'), full_page=True)
    page.locator('#source-gallery').click()
    expect(page.locator('#source-gallery')).to_have_attribute('aria-selected', 'true')
    expect(page.locator('#gallery-panel')).to_be_visible()
    expect(page.locator('[data-action="capture"]')).not_to_be_visible()
    page.wait_for_timeout(200)
    page.screenshot(path=str(out / 'gallery.png'), full_page=True)
    checks.append('Kit scan controls preserve the unavailable-camera state and separate Gallery upload panel')

    for width in [320, 390, 768, 1440]:
        page.set_viewport_size({'width': width, 'height': 900})
        for screen in ['home', 'scan', 'history']:
            page.evaluate('(name) => showScreen(name)', screen)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, screen)
        for button in page.locator('.bottom-nav button').all():
            bounds = button.bounding_box()
            assert bounds['width'] >= 44 and bounds['height'] >= 44
    checks.append('Home, Gallery and Insights fit 320/390/768/1440px widths; navigation meets 44px tap targets')
    for width in [320, 390, 522, 768, 1440]:
        page.set_viewport_size({'width': width, 'height': 1383})
        page.evaluate('() => { showScreen("scan"); selectScanSource("camera"); }')
        tabs = page.locator('#scan-screen > .segmented').bounding_box()
        frame = page.locator('#camera-frame').bounding_box()
        assert abs(frame['x'] - tabs['x']) <= 1, (width, tabs, frame)
        assert abs(frame['width'] - tabs['width']) <= 1, (width, tabs, frame)
        assert abs(frame['x'] + frame['width'] / 2 - width / 2) <= 1, (width, frame)
        shutter = page.locator('[data-action="capture"]').bounding_box()
        assert abs(shutter['x'] + shutter['width'] / 2 - width / 2) <= 1, (width, shutter)
        assert page.evaluate('document.documentElement.scrollHeight <= innerHeight + 1'), width
        for screen in ['home', 'history']:
            page.evaluate('(name) => showScreen(name)', screen)
            assert page.evaluate('document.documentElement.scrollHeight <= innerHeight + 1'), (width, screen)
    page.set_viewport_size({'width': 1440, 'height': 1383})
    page.evaluate('() => { showScreen("scan"); selectScanSource("camera"); }')
    page.wait_for_timeout(200)
    page.screenshot(path=str(out / 'camera-centered-desktop.png'), full_page=True)
    checks.append('Camera frame aligns with source tabs and viewport center, including its 405px height limit; short views have no extra page scroll on tall viewports')
    assert not failed_assets, failed_assets
    assert not errors, errors
    browser.close()

(out / 'checks.json').write_text(json.dumps({'checks': checks, 'javascript_errors': errors,
                                          'failed_assets': failed_assets, 'history_data': 'Isolated browser fixtures'}, indent=2))
print(json.dumps(checks, indent=2))
