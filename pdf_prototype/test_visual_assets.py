"""Offline tests: original pixels, reviewed placements, continuation and EPUB assets."""
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zipfile

import pymupdf
from bs4 import BeautifulSoup
from editions import make_epub
from visual_assets import TABLE_PLACEHOLDER, add_visuals, validate_manifest, html_for_format, digest


class VisualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pdf = self.root / 'original.pdf'
        with pymupdf.open() as d:
            p = d.new_page(width=300, height=400)
            p.insert_text((30, 45), 'Table 1. Original values')
            p.insert_text((30, 80), 'A     22.300')
            p.insert_text((30, 100), 'Footnote: unchanged')
            p = d.new_page(width=300, height=400)
            p.insert_text((30, 60), 'B     19.230')
            d.save(self.pdf)
        self.chapters = [{'id': 'results', 'title': 'Results', 'source_pages': [], 'speech': 'Unchanged narration.',
                          'html': '<p>Before the table.</p><p>Table 1. Simple caption.</p><p>' + TABLE_PLACEHOLDER + '</p><p>After the table.</p>'}]
        self.manifest = {'version': 1, 'pdf_sha256': digest(self.pdf.read_bytes()), 'rewrite_sha256': 'rewrite',
                         'review': 'Inspected fixture.', 'require_all_native_captions': True, 'require_all_placeholders': True,
                         'visuals': [{'id': 'table-1', 'kind': 'table', 'label': 'Table 1', 'caption': 'Original values',
                             'chapter': 'results', 'anchor_prefix': 'Table 1.', 'placeholders_after': 1,
                             'parts': [{'page': 1, 'rect': [20, 25, 250, 115], 'checks': ['22.300', 'Footnote']},
                                       {'page': 2, 'rect': [20, 35, 250, 80], 'checks': ['19.230']}]}]}

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return add_visuals(self.pdf, 'rewrite', self.chapters, self.root, self.manifest)

    def test_crops_equal_source_pixels_and_keep_all_parts(self):
        visuals, _ = self.build()
        self.assertEqual(len(visuals[0]['parts']), 2)
        with pymupdf.open(self.pdf) as source:
            for part in visuals[0]['parts']:
                expected = source[part['page'] - 1].get_pixmap(dpi=300, clip=pymupdf.Rect(part['rect']), alpha=False)
                actual = pymupdf.Pixmap(str(self.root / part['image']))
                self.assertEqual(actual.samples, expected.samples)
                with pymupdf.open(self.root / part['pdf']) as vector:
                    self.assertEqual(len(vector), 1)
                    self.assertAlmostEqual(vector[0].rect.width, part['rect'][2] - part['rect'][0])

    def test_placement_is_at_caption_not_at_mention(self):
        self.chapters[0]['html'] = '<p>Table 1 shows our findings.</p>' + self.chapters[0]['html']
        self.build()
        soup = BeautifulSoup(self.chapters[0]['html'], 'html.parser')
        figures = soup.find_all('figure')
        self.assertEqual(len(figures), 2)
        self.assertEqual(figures[0].find_previous_sibling().get_text(), 'Table 1. Simple caption.')
        self.assertEqual(figures[-1].find_next_sibling().get_text(), 'After the table.')
        self.assertNotIn(TABLE_PLACEHOLDER, self.chapters[0]['html'])
        self.assertEqual(self.chapters[0]['speech'], 'Unchanged narration.')

    def test_wrong_pdf_or_rewrite_is_refused(self):
        for key in ['pdf_sha256', 'rewrite_sha256']:
            m = copy.deepcopy(self.manifest)
            m[key] = 'wrong'
            with self.assertRaises(ValueError):
                validate_manifest(m, self.pdf, 'rewrite', self.chapters)

    def test_bad_crop_or_missing_marker_is_refused(self):
        for rect in [[0, 0, 500, 500], [10, 10, 10, 10], [0, 0, float('nan'), 20], [20, 25, 250, 65]]:
            m = copy.deepcopy(self.manifest)
            m['visuals'][0]['parts'][0]['rect'] = rect
            with self.assertRaises(ValueError):
                validate_manifest(m, self.pdf, 'rewrite', self.chapters)

    def test_missing_or_duplicate_anchor_is_refused(self):
        for text in ['<p>No caption</p>', self.chapters[0]['html'] + '<p>Table 1. Duplicate</p>']:
            chapters = copy.deepcopy(self.chapters)
            chapters[0]['html'] = text
            with self.assertRaises(ValueError):
                validate_manifest(self.manifest, self.pdf, 'rewrite', chapters)

    def test_missing_placeholder_fails_before_writing_assets(self):
        self.chapters[0]['html'] = self.chapters[0]['html'].replace(TABLE_PLACEHOLDER, 'A real paragraph, never delete it.')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.root / 'assets').exists())

    def test_duplicate_ids_and_reverse_continuation_are_refused(self):
        m = copy.deepcopy(self.manifest)
        m['visuals'].append(copy.deepcopy(m['visuals'][0]))
        with self.assertRaises(ValueError):
            validate_manifest(m, self.pdf, 'rewrite', self.chapters)
        m = copy.deepcopy(self.manifest)
        m['visuals'][0]['parts'].reverse()
        with self.assertRaises(ValueError):
            validate_manifest(m, self.pdf, 'rewrite', self.chapters)

    def test_format_adapters_and_epub_package(self):
        visuals, _ = self.build()
        self.assertNotIn('class="visual-image"', html_for_format(self.chapters[0]['html'], 'pdf'))
        self.assertIn('visual-table-1-part-1.xhtml', html_for_format(self.chapters[0]['html'], 'epub'))
        edition = {'id': 'rewrite', 'title': 'Sample', 'original_title': 'Original', 'authors': ['Author'],
                   'year': '2024', 'doi': '10.test/example', 'license': 'CC BY', 'source_page_count': 0,
                   'chapters': self.chapters, 'visuals': visuals}
        make_epub(edition, self.root)
        with zipfile.ZipFile(self.root / 'paper.epub') as archive:
            for part in visuals[0]['parts']:
                self.assertIn('EPUB/' + part['image'], archive.namelist())
                self.assertIn('EPUB/visual-' + part['id'] + '.xhtml', archive.namelist())
            for name in archive.namelist():
                if name.endswith(('.xhtml', '.xml', '.opf')):
                    ET.fromstring(archive.read(name))


if __name__ == '__main__':
    unittest.main()
