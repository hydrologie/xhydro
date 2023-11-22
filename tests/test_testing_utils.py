import xhydro as xh
import xhydro.testing.utils as xhu


def test_publish_release_notes():
    changelog = xhu.publish_release_notes(style="md")

    assert changelog.startswith("# Changes")
    version = xh.__version__
    vsplit = version.split(".")

    v_4history = (
        vsplit[0]
        + "."
        + str(int(vsplit[1]) + 1 if vsplit[2] != "0" else vsplit[1])
        + ".0"
    )
    assert f"## v{v_4history}" in changelog
    assert ":user:`" not in changelog
    assert ":issue:`" not in changelog
    assert ":pull:`" not in changelog

    history_rst = xhu.publish_release_notes(style="rst")
    assert history_rst.startswith("=========\nChangelog\n=========")
