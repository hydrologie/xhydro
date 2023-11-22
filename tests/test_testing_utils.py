import xhydro as xh
import xhydro.testing.utils as xhu


def test_publish_release_notes():
    changelog_md = xhu.publish_release_notes(style="md")

    assert changelog_md.startswith("# Changelog")
    version = xh.__version__
    vsplit = version.split(".")

    v_4history = (
        vsplit[0]
        + "."
        + str(int(vsplit[1]) + 1 if vsplit[2] != "0" else vsplit[1])
        + ".0"
    )
    assert f"## v{v_4history}" in changelog_md
    assert ":user:`" not in changelog_md
    assert ":issue:`" not in changelog_md
    assert ":pull:`" not in changelog_md

    changelog_rst = xhu.publish_release_notes(style="rst")
    assert changelog_rst.startswith("=========\nChangelog\n=========")
