"""
Things that must stay true for the site to survive being deployed.

PLAIN ENGLISH
    The chat assistant runs on a rented computer somewhere. That computer is set up
    differently from a laptop in two ways that have already broken it once each:

      1. It only installs the libraries we declared. Anything we forgot to declare is
         simply missing, and the site fails on its first visitor.
      2. Its disk is read-only apart from one scratch folder. Any code that writes a
         file where it likes will crash.

    These tests check both on a laptop, where the answer is cheap, instead of finding out
    from a broken demo.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _runtime_imports() -> set[str]:
    """Every third-party package the web service loads when it starts."""
    for name in [m for m in sys.modules if m.startswith("clinikit")]:
        del sys.modules[name]

    before = set(sys.modules)
    import clinikit.api.main  # noqa: F401
    loaded = set(sys.modules) - before

    stdlib = set(sys.stdlib_module_names)
    return {
        m.split(".")[0] for m in loaded
        if not m.startswith("_")
        and m.split(".")[0] not in stdlib
        and not m.startswith("clinikit")
    }


def test_the_web_service_does_not_drag_in_the_terminal_ui():
    """
    `rich` draws coloured boxes in a console. The web service has no console.

    It used to be loaded anyway, because the service imported the example messages from
    cli.py and got the whole terminal program with them. They live in examples.py now.
    """
    assert "rich" not in _runtime_imports()


def test_the_web_service_does_not_drag_in_the_machine_learning_stack():
    """
    Part 2 is a separate exercise. pandas, scikit-learn and matplotlib together are
    hundreds of megabytes, and the chat assistant never calls any of them.
    """
    pulled = _runtime_imports()
    for heavy in ("pandas", "numpy", "sklearn", "matplotlib", "joblib", "IPython"):
        assert heavy not in pulled, f"the web service should not need {heavy}"


def test_every_runtime_import_is_declared_for_the_host():
    """
    A package that is imported but not declared works on a laptop -- where it happens to
    be installed for some other reason -- and is missing on the host.
    """
    declared_text = (ROOT / "pyproject.toml").read_text()

    # What we asked the host to install, plus what those packages install in turn.
    installed_in_turn = {
        "annotated_doc", "annotated_types", "anyio", "certifi", "distro", "h11",
        "httpcore", "httpx", "httpx2", "idna", "jiter", "pydantic_core", "sniffio",
        "starlette", "tqdm", "typing_extensions", "typing_inspection",
    }
    aliases = {"dotenv": "python-dotenv", "dateutil": "python-dateutil"}

    for package in _runtime_imports():
        if package in installed_in_turn:
            continue
        name = aliases.get(package, package)
        assert f'"{name}' in declared_text, (
            f"{package!r} is imported at runtime but not declared in pyproject.toml"
        )


def test_an_unwritable_disk_does_not_break_reading_a_message():
    """
    The response cache used to write to the project folder with no error handling. On a
    read-only disk that raised, and the patient lost their reply so that we could fail to
    save a copy of it.
    """
    from clinikit.agent.backends.openai_compat import OpenAICompatibleExtractor
    from clinikit.agent.schema import Extraction, Intent

    extractor = OpenAICompatibleExtractor.__new__(OpenAICompatibleExtractor)
    extractor._use_cache = True
    extractor._key = lambda message, history: "unwritable"

    import clinikit.agent.backends.openai_compat as module
    original = module.CACHE_DIR
    module.CACHE_DIR = Path("/proc/nonexistent/cannot-write-here")
    try:
        # Must not raise.
        extractor._cache_put("hello", (), Extraction(intent=Intent.GREETING, confidence=1.0))
    finally:
        module.CACHE_DIR = original
