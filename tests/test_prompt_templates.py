import unittest


from prompt_templates import fill_subject, split_single_subject_placeholder


class PromptTemplatesTest(unittest.TestCase):
    def test_fill_subject_supports_common_placeholders(self):
        self.assertEqual(fill_subject("{} is a", "Alice"), "Alice is a")
        self.assertEqual(fill_subject("{subject} is a", "Bob"), "Bob is a")
        self.assertEqual(fill_subject("{0} is a", "Carol"), "Carol is a")

    def test_fill_subject_does_not_require_format_safety(self):
        self.assertEqual(
            fill_subject("This has {Des} and {}.", "Y"),
            "This has {Des} and Y.",
        )

    def test_fill_subject_preserves_double_braces(self):
        self.assertEqual(fill_subject("{{Infobox}} {}", "X"), "{{Infobox}} X")

    def test_split_single_subject_placeholder(self):
        prefix, suffix = split_single_subject_placeholder("Hello {Des} {} world")
        self.assertEqual(prefix, "Hello {Des} ")
        self.assertEqual(suffix, " world")

        with self.assertRaises(ValueError):
            split_single_subject_placeholder("no placeholder")


if __name__ == "__main__":
    unittest.main()

