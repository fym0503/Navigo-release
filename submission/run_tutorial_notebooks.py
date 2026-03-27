import argparse
import json
import os
import sys
import traceback
import types
from pathlib import Path


def ensure_ipython_display_stub():
    try:
        import IPython.display  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    ipython_module = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")

    class _Image:
        def __init__(self, data=None, filename=None, url=None, **kwargs):
            self.data = data
            self.filename = filename
            self.url = url
            self.kwargs = kwargs

    class _Markdown:
        def __init__(self, data):
            self.data = data

    def _display(*args, **kwargs):
        del args, kwargs
        return None

    def _get_ipython():
        return None

    display_module.Image = _Image
    display_module.Markdown = _Markdown
    display_module.display = _display
    ipython_module.display = display_module
    ipython_module.get_ipython = _get_ipython
    ipython_module.version_info = (0, 0, 0)

    sys.modules["IPython"] = ipython_module
    sys.modules["IPython.display"] = display_module


def execute_notebook(path: Path, repo_root: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")

    ensure_ipython_display_stub()

    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
    }

    try:
        os.chdir(path.parent)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        if original_pythonpath:
            os.environ["PYTHONPATH"] = str(repo_root) + os.pathsep + original_pythonpath
        else:
            os.environ["PYTHONPATH"] = str(repo_root)

        for idx, cell in enumerate(notebook.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            code = compile(source, f"{path}#cell{idx}", "exec")
            exec(code, namespace)
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath


def parse_args():
    parser = argparse.ArgumentParser(description="Execute tutorial notebooks as plain Python.")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root containing docs/tutorials and the navigo package.",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Notebook path relative to repo root. May be passed multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()

    if args.notebook:
        notebooks = [repo_root / n for n in args.notebook]
    else:
        notebooks = sorted((repo_root / "docs" / "tutorials" / "notebooks").rglob("*.ipynb"))

    failures = []
    for notebook in notebooks:
        print(f"=== Running {notebook.relative_to(repo_root)} ===", flush=True)
        try:
            execute_notebook(notebook, repo_root)
            print(f"=== Completed {notebook.relative_to(repo_root)} ===", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures.append((notebook, exc, traceback.format_exc()))
            print(f"=== Failed {notebook.relative_to(repo_root)} ===", flush=True)
            print(traceback.format_exc(), flush=True)

    if failures:
        print("\nNotebook execution failures:", file=sys.stderr)
        for notebook, exc, _tb in failures:
            print(f"- {notebook.relative_to(repo_root)}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
