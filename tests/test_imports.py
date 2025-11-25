def test_import_package():
    import scripts
    assert hasattr(scripts, "__version__")

def test_import_cli_main():
    import tennisctl
    assert hasattr(tennisctl, "main")
