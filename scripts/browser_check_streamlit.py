"""Run against `streamlit run streamlit_app/app.py` (no Gemini key required)."""
import io
import json
from pathlib import Path

from docx import Document
from playwright.sync_api import sync_playwright, expect


def main():
    output = Path(__file__).resolve().parent.parent / 'docs' / 'screenshots'
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        edge = Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
        browser = playwright.chromium.launch(executable_path=str(edge) if edge.exists() else None, headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto('http://localhost:8501')
        page.get_by_role('button', name='Home office allowance', exact=True).wait_for(timeout=120000)
        page.get_by_role('button', name='Home office allowance', exact=True).click()
        expect(page.locator('[data-testid=stChatMessage]').last).to_contain_text('$750', timeout=30000)
        expect(page.get_by_text('Sources', exact=True)).to_be_visible()
        page.screenshot(path=str(output / 'streamlit.png'), full_page=True)
        document = Document()
        document.add_paragraph('The Orchid launch date is November 17, 2026.')
        buffer = io.BytesIO()
        document.save(buffer)
        page.locator('input[type=file]').set_input_files({'name': 'Orchid.docx', 'mimeType': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'buffer': buffer.getvalue()})
        page.get_by_role('button', name='Index selected documents', exact=True).click()
        page.get_by_text('Orchid.docx is ready.', exact=True).wait_for(timeout=30000)
        page.wait_for_timeout(1100)
        page.locator('textarea').fill('What is the Orchid launch date?')
        page.locator('textarea').press('Enter')
        expect(page.locator('[data-testid=stChatMessage]').last).to_contain_text('November 17, 2026', timeout=30000)
        expect(page.locator('[data-testid=stChatMessage]').last).to_contain_text('Orchid.docx')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert not errors, errors
        browser.close()
        print(json.dumps({'streamlit_browser': 'passed', 'checks': ['semantic sample answer', 'sources', 'DOCX upload', 'uploaded answer', 'mobile overflow'], 'page_errors': errors}))


if __name__ == '__main__':
    main()
