"""Utilities for testing and releasing xhydro."""

import os
import re
from io import StringIO
from pathlib import Path
from typing import Optional, TextIO, Union


def publish_release_notes(
    style: str = "md", file: Optional[Union[os.PathLike, StringIO, TextIO]] = None
) -> Optional[str]:
    """Format release history in Markdown or ReStructuredText.

    Parameters
    ----------
    style: {"rst", "md"}
      Use ReStructuredText formatting or Markdown. Default: Markdown.
    file: {os.PathLike, StringIO, TextIO}, optional
      If provided, prints to the given file-like object. Otherwise, returns a string.

    Returns
    -------
    str, optional

    Notes
    -----
    This function exists solely for development purposes.
    Adapted from xclim.testing.utils.publish_release_notes.
    """
    history_file = Path(__file__).parent.parent.parent.joinpath("HISTORY.rst")

    if not history_file.exists():
        raise FileNotFoundError("History file not found in xhydro file tree.")

    with open(history_file) as hf:
        history = hf.read()

    if style == "rst":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/hydrologie/xhydro/issues/\1>`_",
            r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/hydrologie/xhydro/pull/\>`_",
            r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
        }
    elif style == "md":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"[GH/\1](https://github.com/hydrologie/xhydro/issues/\1)",
            r":pull:`([0-9]+)`": r"[PR/\1](https://github.com/hydrologie/xhydro/pull/\1)",
            r":user:`([a-zA-Z0-9_.-]+)`": r"[@\1](https://github.com/\1)",
        }
    else:
        raise NotImplementedError()

    for search, replacement in hyperlink_replacements.items():
        history = re.sub(search, replacement, history)

    if style == "md":
        history = history.replace("=======\nHistory\n=======", "# History")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, history)
            for grouping in found:
                fixed_grouping = (
                    str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
                )
                search = rf"({fixed_grouping})\n([\{level}]{'{' + str(len(grouping[1])) + '}'})"
                replacement = f"{'##' if level=='-' else '###'} {grouping[0]}"
                history = re.sub(search, replacement, history)

        link_expressions = r"[\`]{1}([\w\s]+)\s<(.+)>`\_"
        found = re.findall(link_expressions, history)
        for grouping in found:
            search = rf"`{grouping[0]} <.+>`\_"
            replacement = f"[{str(grouping[0]).strip()}]({grouping[1]})"
            history = re.sub(search, replacement, history)

    if not file:
        return history
    print(history, file=file)
