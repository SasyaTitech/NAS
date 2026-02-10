from __future__ import annotations

from typing import List, Tuple


_PLACEHOLDERS = ("{}", "{subject}", "{0}")


def fill_subject(template: str, subject: str) -> str:
    """
    Replaces subject placeholders in a prompt template without invoking Python's
    `str.format`, so unrelated braces like `{Des}` won't raise KeyError.

    Supported placeholders: `{}`, `{0}`, `{subject}`.
    Escaped brace pairs `{{` and `}}` are preserved verbatim.
    """

    if not isinstance(template, str):
        template = str(template)
    if not isinstance(subject, str):
        subject = str(subject)

    out: List[str] = []
    i = 0
    n = len(template)
    while i < n:
        if template.startswith("{{", i):
            out.append("{{")
            i += 2
            continue
        if template.startswith("}}", i):
            out.append("}}")
            i += 2
            continue
        if template.startswith("{}", i):
            out.append(subject)
            i += 2
            continue
        if template.startswith("{0}", i):
            out.append(subject)
            i += 3
            continue
        if template.startswith("{subject}", i):
            out.append(subject)
            i += 9
            continue

        out.append(template[i])
        i += 1

    return "".join(out)


def split_single_subject_placeholder(template: str) -> Tuple[str, str]:
    """
    Splits a template into (prefix, suffix) around exactly one supported subject placeholder.

    Supported placeholders: `{}`, `{0}`, `{subject}`.
    Escaped brace pairs `{{` and `}}` are ignored (treated as literals).
    """

    if not isinstance(template, str):
        template = str(template)

    matches: List[Tuple[int, int]] = []
    i = 0
    n = len(template)
    while i < n:
        if template.startswith("{{", i) or template.startswith("}}", i):
            i += 2
            continue

        for ph in _PLACEHOLDERS:
            if template.startswith(ph, i):
                matches.append((i, len(ph)))
                i += len(ph)
                break
        else:
            i += 1

    if len(matches) != 1:
        raise ValueError(
            "Context template must contain exactly one subject placeholder "
            f"from {_PLACEHOLDERS}; got {len(matches)} for template={template!r}"
        )

    start, length = matches[0]
    return template[:start], template[start + length :]

