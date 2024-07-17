"""Translate missing msgstr entries in .po files using the specified translator."""

import re
import time
from glob import glob
from pathlib import Path

import deep_translator


def translate_missing_po_entries(
    dir_path: str,
    translator: str = "GoogleTranslator",
    source_lang: str = "en",
    target_lang: str = "fr",
    clean_old_entries: bool = False,
    **kwargs,
):
    r"""
    Translate missing msgstr entries in .po files using the specified translator.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing the .po files.
    translator : str
        The translator to use. Uses GoogleTranslator by default, but can be changed to any other translator supported by `deep_translator`.
    source_lang : str
        The source language of the .po files. Defaults to "en".
    target_lang : str
        The target language of the .po files. Defaults to "fr".
    clean_old_entries : bool
        Whether to clean old entries in the .po files. Defaults to False.
    \*\*kwargs : dict
        Additional keyword arguments to pass to the translator.
    """
    # FIXME: We'll need to add a way to detect the entries flagged "fuzzy" and re-translate them

    # Define regex patterns for msgid and msgstr
    msgid_pattern = re.compile(r'msgid (.*?)(?=(msgstr "|$))', re.DOTALL)
    msgstr_pattern = re.compile(r"msgstr (.*?)(?=(#~|#:|$))")

    # Initialize the translator
    translator = getattr(deep_translator, translator)(
        source=source_lang, target=target_lang, **kwargs
    )

    # Get all .po files
    files = glob(f"{dir_path}/**/*.po", recursive=True)

    number_of_calls = 0
    for file_path in files:
        if not any(
            dont_translate in file_path for dont_translate in ["changelog", "apidoc"]
        ):
            with open(file_path, "r+", encoding="utf-8") as file:
                content = file.read()

                # Find all msgid and msgstr pairs
                msgids = msgid_pattern.findall(str(content))
                msgids_to_translate = []
                for i, msgid in enumerate(msgids):
                    msgids[i] = msgid[0] if msgid[0] != '""\n' else ""
                    msgids_to_translate.extend(
                        [msgid[0].replace("\n", "").replace('"', "")]
                    )
                msgstrs = msgstr_pattern.findall(str(content).replace("\n", ""))
                for i, msgstr in enumerate(msgstrs):
                    msgstrs[i] = msgstr[0].replace('"', "")

                # Track if the file was modified
                modified = False

                for msgid, msgid_to_translate, msgstr in zip(
                    msgids, msgids_to_translate, msgstrs
                ):
                    # Check if translation is missing
                    if msgid and not msgstr:
                        # Translate the missing string
                        translated_text = translator.translate(msgid_to_translate)

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
                            f'msgid {msgid}msgstr ""',
                            f'msgid {msgid}msgstr "{translated_text}"',
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
                    file.seek(0)
                    file.write(content)
                    file.truncate()


# FIXME: Add argparse to make it a command-line tool
if __name__ == "__main__":
    dir_path = Path(__file__).parents[1] / "docs" / "locales" / "fr" / "LC_MESSAGES"
    translate_missing_po_entries(dir_path)
