import xhydro as xh


# Smoke test for xscen functions that are imported into xhydro
def test_xscen_imported():
    assert callable(getattr(xh.utils, "health_checks"))


def test_publish_release_notes():
    history = xh.utils.publish_release_notes(style="md")

    assert history.startswith("# History")
    version = xh.__version__
    vsplit = version.split(".")

    v_4history = (
        vsplit[0]
        + "."
        + str(int(vsplit[1]) + 1 if vsplit[2] != "0" else vsplit[1])
        + ".0"
    )
    assert f"## v{v_4history}" in history
    if vsplit[2] != "0":
        assert "(unreleased)" in history
    else:
        assert "(unreleased)" not in history
    assert ":user:`" not in history
    assert ":issue:`" not in history
    assert ":pull:`" not in history

    history_rst = xh.utils.publish_release_notes(style="rst")
    assert history_rst.startswith("=======\nHistory\n=======")
