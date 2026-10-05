import unittest
import xml.etree.ElementTree as ET

from measure import article_text, normalize, render


class ExtractionTests(unittest.TestCase):
    def text(self, xml):
        return normalize(render(ET.fromstring(xml)))

    def test_inline_words_are_not_split(self):
        self.assertEqual(self.text('<p>pre<italic>print</italic> text</p>'), 'preprint text')

    def test_table_alternative_beats_empty_graphic(self):
        text = self.text('<alternatives><graphic/><table><tr><td>Cell one</td><td>Cell two</td></tr></table></alternatives>')
        self.assertEqual(text, 'Cell one Cell two')

    def test_equation_alternatives_counted_once(self):
        text = self.text('<alternatives><graphic/><math><mi>x</mi></math><tex-math>x^2</tex-math></alternatives>')
        self.assertEqual(text, 'x^2')

    def test_bibliography_removed_but_not_acknowledgments(self):
        root = ET.fromstring('<article><body><p>Main text.</p></body><back><ack>Thanks.</ack><ref-list><ref>Cited paper.</ref></ref-list></back></article>')
        self.assertIn('Cited paper.', article_text(root, True))
        self.assertNotIn('Cited paper.', article_text(root, False))
        self.assertIn('Thanks.', article_text(root, False))
        self.assertIsNotNone(root.find('.//ref-list'))

    def test_publisher_object_identifier_excluded(self):
        self.assertEqual(self.text('<fig><object-id>doi:test</object-id><caption>Caption.</caption></fig>'), 'Caption.')

    def test_paragraphs_separated(self):
        self.assertEqual(self.text('<body><p>First.</p><p>Second.</p></body>'), 'First.\n\nSecond.')


if __name__ == '__main__':
    unittest.main()
