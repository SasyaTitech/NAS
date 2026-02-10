import unittest

from dsets.wikibigedit import _ensure_single_placeholder_template


class WikiBigEditTemplateTest(unittest.TestCase):
    def test_existing_placeholder_is_preserved(self):
        self.assertEqual(
            _ensure_single_placeholder_template("Hello {} world", "Alice"),
            "Hello {} world",
        )

    def test_fuzzy_subject_match_with_punctuation(self):
        subject = "Baroness Fritchie also known as Rennie Fritchie"
        prompt = "What is the family name of Baroness Fritchie, also known as Rennie Fritchie?"
        templ = _ensure_single_placeholder_template(prompt, subject)
        self.assertEqual(templ.count("{}"), 1)

    def test_fallback_when_subject_not_found(self):
        subject = "CompletelyDifferent"
        prompt = "What is the family name of Baroness Fritchie?"
        templ = _ensure_single_placeholder_template(prompt, subject)
        self.assertTrue(templ.startswith("{} "))
        self.assertEqual(templ.count("{}"), 1)


if __name__ == "__main__":
    unittest.main()

