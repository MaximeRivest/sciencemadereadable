"""Check e-ink pagination, stored position, audio loading and downloadable formats."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
from threading import Thread
import zipfile

import pymupdf
from playwright.sync_api import sync_playwright


def check(folder):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_head(self):
            self.remaining = None
            path = Path(self.translate_path(self.path))
            range_header = self.headers.get('Range')
            if path.suffix == '.mp3' and path.is_file() and range_header:
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', range_header)
                if match:
                    size = path.stat().st_size
                    start, end = int(match[1]), min(int(match[2]) if match[2] else size - 1, size - 1)
                    if start >= size or end < start:
                        self.send_error(416)
                        return None
                    stream = path.open('rb')
                    stream.seek(start)
                    self.remaining = end - start + 1
                    self.send_response(206)
                    self.send_header('Content-Type', 'audio/mpeg')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                    self.send_header('Content-Length', str(self.remaining))
                    self.end_headers()
                    return stream
            return super().send_head()

        def copyfile(self, source, output):
            try:
                if self.remaining is None:
                    return super().copyfile(source, output)
                while self.remaining > 0:
                    data = source.read(min(65536, self.remaining))
                    if not data:
                        break
                    output.write(data)
                    self.remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser cancelled an old track after switching chapters
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(folder)))
    Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=os.environ.get('CHROMIUM_BIN') or shutil.which('chromium'), headless=True)
            page = browser.new_page(viewport={'width': 1080, 'height': 1440}, device_scale_factor=1)
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='networkidle')
            assert page.locator('#home-title').inner_text()
            page.screenshot(path=str(folder.parent / 'reader-library.png'), full_page=True)
            has_visuals = page.locator('#visual-grid button').count() > 0
            if has_visuals:
                assert page.locator('#visual-grid button').count() == 6
                page.locator('#visual-grid button').first.click()
                page.wait_for_function("document.getElementById('visual-full').complete && document.getElementById('visual-full').naturalWidth > 1000")
                assert 'part 1 of 2' in page.locator('#visual-title').inner_text()
                before = page.locator('#visual-full').bounding_box()['width']
                page.click('#zoom-in')
                assert page.locator('#visual-full').bounding_box()['width'] > before
                page.click('#visual-next')
                assert 'part 2 of 2' in page.locator('#visual-title').inner_text()
                assert page.request.get(page.locator('#visual-pdf').evaluate('a => a.href')).ok
                page.click('#visual-next')
                page.wait_for_function("document.getElementById('visual-full').complete && document.getElementById('visual-full').naturalWidth > 0")
                page.screenshot(path=str(folder.parent / 'reader-original-figure.png'), full_page=False)
                page.click('#visual-close')
            page.click('#start')
            page.wait_for_timeout(150)
            first = page.locator('#progress').inner_text()
            assert '1 /' in first
            page.click('#next')
            second = page.locator('#progress').inner_text()
            assert second != first
            assert page.evaluate("document.querySelector('#viewport').scrollLeft > 0")
            page.reload(wait_until='networkidle')
            page.click('#start')
            page.wait_for_timeout(150)
            assert page.locator('#progress').inner_text() == second
            assert page.evaluate("JSON.parse(localStorage.getItem(key)).page >= 0")
            page.select_option('#chapter', '2')
            page.wait_for_timeout(150)
            assert 'Methods' in page.locator('#progress').inner_text()
            page.click('#settings-button')
            page.locator('#font-size').fill('28')
            page.locator('#font-size').dispatch_event('change')
            page.select_option('#line-space', '1.85')
            page.wait_for_timeout(150)
            assert page.evaluate("getComputedStyle(document.getElementById('flow')).fontSize") == '28px'
            page.click('#settings-button')
            page.wait_for_timeout(150)
            page.screenshot(path=str(folder.parent / 'reader-tablet.png'), full_page=True)
            for size in [{'width': 390, 'height': 844}, {'width': 1440, 'height': 900}]:
                page.set_viewport_size(size)
                page.wait_for_timeout(250)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert page.evaluate('document.documentElement.scrollHeight <= innerHeight + 1')
                count = int(page.locator('#progress').inner_text().split('/')[1].split('·')[0].strip())
                assert count > 1
                page.click('#next')
                # Each displayed page must contain some visible text, not just an empty column.
                visible = page.evaluate("""() => {const v=document.getElementById('viewport').getBoundingClientRect(); const r=document.createRange();let visible=0; const walk=document.createTreeWalker(document.getElementById('flow'),NodeFilter.SHOW_TEXT);while(walk.nextNode()){if(!walk.currentNode.textContent.trim())continue;r.selectNodeContents(walk.currentNode);for(const b of r.getClientRects())if(b.right>v.left&&b.left<v.right&&b.bottom>v.top&&b.top<v.bottom)visible++;}return visible;}""")
                assert visible > 0
                if has_visuals:
                    page.evaluate("""() => {const f=document.querySelector('#flow .source-visual'); const v=document.getElementById('viewport'); const x=f.getBoundingClientRect().left-v.getBoundingClientRect().left+v.scrollLeft;updatePage(Math.floor((x+2)/stride));}""")
                    bounds = page.evaluate("""() => {const f=document.querySelector('#flow .source-visual').getBoundingClientRect(),v=document.getElementById('viewport').getBoundingClientRect();return {left:f.left-v.left,right:f.right-v.right,top:f.top-v.top,bottom:f.bottom-v.bottom};}""")
                    assert abs(bounds['left']) < 2 and bounds['right'] <= 2 and bounds['top'] >= -2 and bounds['bottom'] <= 2, bounds
                    page.screenshot(path=str(folder.parent / ('reader-inline-' + str(size['width']) + '.png')), full_page=True)
                    page.locator('#flow .source-visual a.visual-image').first.click()
                    assert page.locator('#visual-dialog').is_visible()
                    page.click('#zoom-in')
                    assert page.locator('#visual-full').bounding_box()['width'] > page.locator('#zoom-stage').bounding_box()['width']
                    page.click('#visual-close')
            expected_track = str(int(page.locator('#chapter').input_value()) + 1)  # first audio track is the notice
            page.click('#listen-here')
            page.wait_for_function("Number.isFinite(document.getElementById('player').duration) && document.getElementById('player').duration > 0")
            assert page.locator('#track').input_value() == expected_track
            page.wait_for_function("document.getElementById('player').readyState >= 3 && document.getElementById('player').seekable.length > 0")
            page.evaluate("document.getElementById('player').pause()")
            page.click('#forward-audio')
            page.wait_for_function("document.getElementById('player').currentTime >= 14")
            page.select_option('#track', '1')
            page.wait_for_function("Number.isFinite(document.getElementById('player').duration) && document.getElementById('player').duration > 0")
            for name in ['paper.epub', 'paper.pdf', 'paper.mp3', 'reading.zip', 'narration.txt']:
                assert page.request.get(f'http://127.0.0.1:{server.server_port}/{name}').ok
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
    with pymupdf.open(folder / 'paper.pdf') as doc, pymupdf.open(folder / 'original.pdf') as original:
        assert len(doc) > len(original)
        assert doc.get_toc()
        assert 'simplified' in doc[0].get_text().lower()
        offset = len(doc) - len(original)
        for i in range(len(original)):
            assert doc[offset + i].get_text() == original[i].get_text()
            appended = doc[offset + i].get_pixmap(matrix=pymupdf.Matrix(.3, .3))
            source = original[i].get_pixmap(matrix=pymupdf.Matrix(.3, .3))
            assert (appended.width, appended.height) == (source.width, source.height)
            # A combined PDF can use a different document-level colour profile.
            # Require matching geometry/text and only tiny channel differences.
            delta = [abs(a - b) for a, b in zip(appended.samples, source.samples)]
            assert max(delta) <= 5 and sum(delta) / len(delta) < 1
    if (folder / 'visuals.json').exists():
        audit = json.loads((folder / 'visuals.json').read_text())
        with pymupdf.open(folder / 'original.pdf') as original:
            for visual in audit['visuals']:
                for part in visual['parts']:
                    expected = original[part['page'] - 1].get_pixmap(dpi=part['dpi'], clip=pymupdf.Rect(part['rect']), alpha=False)
                    actual = pymupdf.Pixmap(str(folder / part['image']))
                    assert expected.samples == actual.samples
        with pymupdf.open(folder / 'paper.pdf') as doc:
            source_pages = [p for p in doc if 'Original labels, values and caption preserved' in p.get_text()]
            assert len(source_pages) == 8
            assert all(round(p.rect.width) > 500 for p in source_pages)
            assert any('22.300' in p.get_text() for p in source_pages), 'Original table cells must remain PDF text, not just a raster picture'
    audio = json.loads((folder / 'audio.json').read_text())
    for track in audio['tracks'] + [{'file': 'paper.mp3', 'seconds': audio['seconds']}]:
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(folder / track['file'])]))
        assert probe['streams'][0]['codec_name'] == 'mp3'
        assert abs(float(probe['format']['duration']) - track['seconds']) < .2
    with zipfile.ZipFile(folder / 'reading.zip') as archive:
        assert archive.testzip() is None
        assert {'index.html', 'paper.epub', 'paper.pdf', 'paper.mp3'}.issubset(archive.namelist())
    print('Reader checked: pagination, fonts, resume, inline originals, zoom/pan viewer, original crop pixels, vector PDF tables, chapter audio, downloads, source appendix, MP3 durations, offline ZIP. No script errors.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, default=Path('reading'), nargs='?')
    check(parser.parse_args().folder.resolve())
