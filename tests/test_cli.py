import pytest

from apt_rag.cli import main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_help_command_lists_commands(capsys):
    assert main(["help"]) == 0
    out = capsys.readouterr().out
    assert "inspect" in out and "search" in out and "help" in out


def test_help_command_for_subcommand(capsys):
    assert main(["help", "search"]) == 0
    assert "--top-n" in capsys.readouterr().out


def test_help_command_unknown_topic(capsys):
    assert main(["help", "nope"]) == 1
    assert "nope" in capsys.readouterr().err
