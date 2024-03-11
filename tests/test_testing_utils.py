from pathlib import Path

import pytest

import xhydro.testing.utils as xhu


@pytest.mark.requires_docs
def test_publish_release_notes(tmp_path):
    temp_md_filename = tmp_path.joinpath("version_info.md")
    xhu.publish_release_notes(
        style="md",
        file=temp_md_filename,
        changes=Path(__file__).parent.parent.joinpath("CHANGES.rst"),
    )

    with open(temp_md_filename) as f:
        changelog = f.read()

    assert changelog.startswith("# Changelog")
    assert ":user:`" not in changelog
    assert ":issue:`" not in changelog
    assert ":pull:`" not in changelog

    temp_rst_filename = tmp_path.joinpath("version_info.rst")
    xhu.publish_release_notes(
        style="rst",
        file=temp_rst_filename,
        changes=Path(__file__).parent.parent.joinpath("CHANGES.rst"),
    )
    with open(temp_rst_filename) as f:
        changelog_rst = f.read()
    assert changelog_rst.startswith("=========\nChangelog\n=========")
