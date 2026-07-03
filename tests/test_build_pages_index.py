import unittest

from scripts import build_pages_index as report


class PagesIndexNavigationTest(unittest.TestCase):
    def test_619_batch_matrix_is_prominent(self) -> None:
        filename = "vllm_standalone_results_20260619.html"

        primary_links = report.primary_links_html()
        report_hub = report.report_hub_html()
        visible_hub, separator, collapsed_hub = report_hub.partition(
            '<details class="archive-links">'
        )

        self.assertIn(f'href="reports/{filename}"', primary_links)
        self.assertTrue(separator)
        self.assertIn(f'href="reports/{filename}"', visible_hub)
        self.assertNotIn(f'href="reports/{filename}"', collapsed_hub)


if __name__ == "__main__":
    unittest.main()
