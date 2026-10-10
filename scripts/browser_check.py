"""Run against a local server. Uses installed Edge, or Playwright Chromium."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'docs' / 'screenshots'
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    with sync_playwright() as p:
        edge = Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
        browser = p.chromium.launch(executable_path=str(edge) if edge.exists() else None, headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto('http://127.0.0.1:8000', wait_until='networkidle')
        expect(page.locator('#document-count')).to_have_text('3')
        page.screenshot(path=str(OUTPUT / 'workspace.png'), full_page=True)
        page.get_by_role('button', name='What’s the home office allowance?').click()
        page.wait_for_selector('.assistant-message')
        assert '$750' in page.locator('.answer-text').inner_text()
        page.locator('.citation-link').first.click()
        assert page.locator('.passage.highlight').count() == 1
        page.screenshot(path=str(OUTPUT / 'answer.png'), full_page=True)
        page.get_by_role('button', name='Open source').first.click()
        page.locator('#source-dialog').wait_for(state='visible')
        assert page.locator('#source-dialog').is_visible()
        page.get_by_role('button', name='Close source').click()
        page.locator('#upload').set_input_files({'name': 'Browser-test.txt', 'mimeType': 'text/plain', 'buffer': b'The amber telescope observation window begins at 21:30 UTC.'})
        expect(page.locator('#document-count')).to_have_text('4')
        page.wait_for_timeout(1100)  # Respect the demo's one-second chat throttle.
        page.locator('#question').fill('When does the amber telescope observation window begin?')
        page.locator('#send').click()
        expect(page.locator('.assistant-message')).to_have_count(2)
        assert '21:30' in page.locator('.answer-text').last.inner_text()
        page.reload(wait_until='networkidle')
        expect(page.locator('.assistant-message')).to_have_count(2)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('() => document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(OUTPUT / 'mobile.png'), full_page=True)
        page.on('dialog', lambda dialog: dialog.accept())
        page.get_by_role('button', name='Delete Browser-test.txt').click()
        expect(page.locator('#document-count')).to_have_text('3')
        assert page.locator('.assistant-message').count() == 0
        assert not errors, errors
        browser.close()
        print(json.dumps({'browser': 'passed', 'checks': ['sample answer', 'citation', 'source dialog', 'upload', 'uploaded answer', 'history reload', 'mobile overflow', 'delete'], 'page_errors': errors}))


if __name__ == '__main__':
    main()
