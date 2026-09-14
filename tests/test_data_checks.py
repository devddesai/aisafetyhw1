"""Offline regression checks using synthetic dialogue and mocked HTTP responses."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

import scrape_transcripts as scraper
import validate_pairs as validator


def pair(content="Synthetic response", episode="synthetic"):
    return {
        "messages": [{"role": "user", "content": "Synthetic prompt"},
                     {"role": "assistant", "content": content}],
        "meta": {"episode": episode},
    }


def response(html, failed=False):
    result = Mock(text=html)
    if failed:
        result.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
    return result


class ValidatorTests(unittest.TestCase):
    def test_rejects_invalid_interior_role(self):
        row = pair()
        row["messages"][1:1] = [
            {"role": role, "content": "Synthetic context"}
            for role in ("assistant", "system", "user")
        ]
        errors = []
        validator.check_shape([row], "train", errors)
        self.assertTrue(any("expected user" in error for error in errors))

    def test_accepts_alternating_roles(self):
        errors = []
        validator.check_shape([pair()], "train", errors)
        self.assertEqual(errors, [])

    def test_multiple_catchphrases_count_as_one_response(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            validator.describe([pair("bazinga my spot"), pair()], "synthetic")
        self.assertIn("2 phrase matches in 1/2 responses (50.00% of targets)", output.getvalue())

    def test_empty_splits_fail_with_helpful_message(self):
        for split, rows in (("train", [[], [pair()]]), ("heldout", [[pair()], []])):
            with self.subTest(split=split), patch("sys.argv", ["validate_pairs.py"]), \
                    patch.object(validator, "load", side_effect=rows), \
                    contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    validator.main()
                self.assertNotEqual(raised.exception.code, 0)
                self.assertIn(f"{split} split is empty", errors.getvalue())


class ScraperTests(unittest.TestCase):
    links = ('<a href="https://example.test/one/">Series 1 Episode 1 Synthetic</a>'
             '<a href="https://example.test/two/">Series 1 Episode 2 Synthetic</a>')
    dialogue = '<div class="entry"><p>Sheldon: Synthetic response for a test.</p></div>'

    def run_scraper(self, output, responses):
        session = Mock()
        session.get.side_effect = responses
        with patch("sys.argv", ["scrape_transcripts.py", "--out", str(output), "--delay", "0"]), \
                patch.object(scraper.requests, "Session", return_value=session), \
                contextlib.redirect_stderr(io.StringIO()):
            scraper.main()

    def test_failed_scrape_preserves_existing_file(self):
        cases = {
            "discovery HTTP error": [response(self.links, failed=True)],
            "no links": [response("<p>No episode links</p>")],
            "episode HTTP error": [response(self.links), response(self.dialogue),
                                   response(self.dialogue, failed=True)],
            "empty episode": [response(self.links), response(self.dialogue), response("<p>No dialogue</p>")],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw_episodes.json"
            for name, responses in cases.items():
                with self.subTest(case=name):
                    output.write_text("existing dataset\n")
                    with self.assertRaises((SystemExit, requests.HTTPError)) as raised:
                        self.run_scraper(output, responses)
                    if isinstance(raised.exception, SystemExit):
                        self.assertNotEqual(raised.exception.code, 0)
                    self.assertEqual(output.read_text(), "existing dataset\n")

    def test_successful_scrape_writes_all_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw_episodes.json"
            self.run_scraper(output, [response(self.links), response(self.dialogue), response(self.dialogue)])
            episodes = json.loads(output.read_text())
            self.assertEqual(len(episodes), 2)
            self.assertTrue(all(len(episode["lines"]) == 1 for episode in episodes))


if __name__ == "__main__":
    unittest.main()
