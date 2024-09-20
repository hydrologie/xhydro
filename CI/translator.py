"""Translate missing msgstr entries in .po files using the specified translator."""

import logging
import re
import time
from pathlib import Path
from typing import Union

import deep_translator

logger = logging.getLogger(__name__)


def translate_missing_po_entries(  # noqa: C901
    dir_path: str | Path,
    translator: str = "GoogleTranslator",
    source_lang: str = "en",
    target_lang: str = "fr",
    clean_old_entries: bool = True,
    overwrite_fuzzy: bool = True,
    **kwargs,
):
    r"""Translate missing msgstr entries in .po files using the specified translator.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing the .po files.
    translator : str
        The translator to use.
        Uses GoogleTranslator by default, but can be changed to any other translator supported by `deep_translator`.
    source_lang : str
        The source language of the .po files. Defaults to "en".
    target_lang : str
        The target language of the .po files. Defaults to "fr".
    clean_old_entries : bool
        Whether to clean old entries in the .po files. Defaults to True.
        These are entries that do not exist anymore in the source code, but are still present in the .po files.
    overwrite_fuzzy : bool
        Whether to overwrite fuzzy entries in the .po files. Defaults to True.
        Fuzzy entries are entries that have been marked as needing review, so they are usually wrong.
    \*\*kwargs : dict
        Additional keyword arguments to pass to the translator.
    """
    msg_pattern = re.compile(r"msgid (.*?)(?=(#~|#:|$))", re.DOTALL)
    fuzzy_pattern = re.compile(r"#, fuzzy(.*?)\nmsgid (.*?)(?=(#~|#:|$))", re.DOTALL)

    # Initialize the translator
    translator = getattr(deep_translator, translator)(
        source=source_lang, target=target_lang, **kwargs
    )

    # Get all .po files
    files = [
        f
        for f in Path(dir_path).rglob("*.po")
        if not any(d in str(f) for d in ["changelog", "apidoc"])
    ]

    number_of_calls = 0
    for file_path in files:
        with Path(file_path).open("r+", encoding="utf-8") as file:
            content = file.read()

            # Find all fuzzy entries
            fuzzy_entries = fuzzy_pattern.findall(str(content))
            if len(fuzzy_entries) > 0 and overwrite_fuzzy:
                msg = f"Found {len(fuzzy_entries)} fuzzy entries in {file_path}"
                logger.info(msg)
                for i in fuzzy_entries:
                    entry = i[1].split("\nmsgstr ")
                    # Remove the fuzzy entry
                    content = content.replace(entry[1], '""\n\n')
                # Since we can't guarantee the exact way the fuzzy entry was written
                # we remove the fuzzy tag in 2 steps
                content = content.replace(", fuzzy", "")
                content = content.replace("#\nmsgid", "msgid")

            # Find all msgid and msgstr pairs
            msgids = []
            msgstrs = []
            content_valid = content.split("#~")[0]  # Skip old entries
            for i in msg_pattern.findall(str(content_valid)):
                ids, strs = i[0].split("\nmsgstr ")
                ids = ids if ids != '""' else ""
                strs = strs.replace('\\"', "'").replace('"', "").replace("\n", "")
                msgids.extend([ids])
                msgstrs.extend([strs])

            # Track if the file was modified
            modified = False

            for msgid, msgstr in zip(msgids, msgstrs):
                # Check if translation is missing
                if msgid and not msgstr:
                    # Translate the missing string
                    translated_text = translator.translate(
                        msgid.replace('\\"', "'").replace('"', "").replace("\n", "")
                    )

                    # Split the translated text into lines of max 60 characters
                    if len(translated_text) > 70:  # 70 to include the spaces
                        words = translated_text.split()
                        length = 0
                        words[0] = '"\n"' + words[0]
                        for i in range(len(words)):
                            length += len(words[i])
                            if length > 60:
                                words[i] = '"\n"' + words[i]
                                length = 0
                        translated_text = " ".join(words)

                    # Replace the empty msgstr with the translated text
                    content = content.replace(
                        f'msgid {msgid}\nmsgstr ""',
                        f'msgid {msgid}\nmsgstr "{translated_text}"',
                        1,
                    )
                    modified = True

                    # Sleep to avoid rate limiting
                    number_of_calls += 1
                    if number_of_calls % 100 == 0:
                        time.sleep(60)
                    else:
                        time.sleep(1)

            if clean_old_entries:
                is_old = str(content).split("#~")
                if len(is_old) > 1:
                    content = is_old[0]
                    modified = True

            # If modifications were made, write them back to the file
            if modified:
                msg = f"Updating translations in {file_path}"
                logger.info(msg)
                file.seek(0)
                file.write(content)
                file.truncate()


# FIXME: Add argparse to make it a command-line tool
if __name__ == "__main__":
    directory = Path(__file__).parents[1] / "docs" / "locales" / "fr" / "LC_MESSAGES"
    translate_missing_po_entries(directory)
