"""Offline tests for export safety and speech text preservation."""
import re
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from editions import clean_chapter, narration, split_speech, make_epub


class EditionTests(unittest.TestCase):
    def test_generated_html_and_markdown_tables_are_removed(self):
        body, count = clean_chapter('Before.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n<table><tr><td>999</td></tr></table>\n\nAfter.')
        self.assertEqual(count, 2)
        self.assertNotIn('<table', body)
        self.assertNotIn('999', body)
        self.assertIn('Before.', body)
        self.assertIn('After.', body)
        self.assertEqual(body.count('Original table:'), 2)

    def test_unsafe_markup_cannot_reach_readers(self):
        body, _ = clean_chapter('<script>alert(1)</script><img src="https://tracking.invalid/x"><p onclick="bad()">Text <a href="javascript:bad()">link</a></p>')
        for bad in ['script', 'tracking.invalid', 'onclick', 'javascript:']:
            self.assertNotIn(bad, body)
        ET.fromstring('<div>' + body + '</div>')

    def test_citation_omission_does_not_drop_measurements(self):
        spoken = narration('<p>In 20 samples [1,2–4], 89.9% to 114.8% was recovered at 0.1 ng g⁻¹.</p>')
        self.assertNotIn('[1', spoken)
        for number in ['20', '89.9%', '114.8%', '0.1']:
            self.assertIn(number, spoken)
        self.assertIn('nanograms per gram', spoken)

    def test_tables_and_equations_are_announced(self):
        body, _ = clean_chapter('Text.\n\n<table><tr><td>8</td></tr></table>\n\n$x=2$')
        spoken = narration(body)
        self.assertIn('A table appears here', spoken)
        self.assertIn('A mathematical expression appears here', spoken)
        self.assertNotIn('x=2', spoken)

    def test_chunking_never_drops_or_duplicates_non_whitespace(self):
        text = ('This is a sentence with 0.1 and 8.3 percent. ' * 60) + '\n\nLast paragraph.'
        chunks = split_speech(text, 250)
        self.assertTrue(all(len(c) <= 250 for c in chunks))
        self.assertEqual(re.sub(r'\s', '', ''.join(chunks)), re.sub(r'\s', '', text))

    def test_long_unbroken_input_fails_instead_of_truncating(self):
        with self.assertRaises(ValueError):
            split_speech('a' * 1200)

    def test_inequalities_remain_inequalities(self):
        text = narration('<p>Less &lt; 0.05, more &gt; 2, spread ± 0.1.</p>')
        self.assertIn('less than 0.05', text)
        self.assertIn('greater than 2', text)
        self.assertIn('plus or minus 0.1', text)

    def test_epub_mimetype_and_xml(self):
        edition = {'id': 'abc', 'title': 'A & B', 'original_title': 'Original', 'authors': ['An Author'],
                   'year': '2024', 'doi': '10.test/example', 'license': 'CC BY', 'source_page_count': 0,
                   'chapters': [{'id': 'abstract', 'title': 'Abstract', 'html': '<p>Text &amp; more.</p>', 'source_pages': []}]}
        with tempfile.TemporaryDirectory() as d:
            make_epub(edition, Path(d))
            with zipfile.ZipFile(Path(d) / 'paper.epub') as z:
                self.assertEqual(z.infolist()[0].filename, 'mimetype')
                self.assertEqual(z.infolist()[0].compress_type, zipfile.ZIP_STORED)
                self.assertEqual(z.read('mimetype'), b'application/epub+zip')
                for name in z.namelist():
                    if name.endswith(('.xml', '.xhtml', '.opf')):
                        ET.fromstring(z.read(name))


if __name__ == '__main__':
    unittest.main()
