import inspect
import subprocess

import pytest


def python_import_as_command(python_code, module_name):
    """Refactors python code for running as terminal commands (python -c "<code>")

    Args:
        python_code: code that must be formatted
        module_name: module that must be imported
    """
    python_code = inspect.cleandoc(python_code).format(MODULE=module_name)
    return python_code.replace("\n", ";")


def run_import_command(code: str, module: str):
    """Runs python import code in terminal"""
    formatted_code = python_import_as_command(code, module)

    # run code in new process
    res = subprocess.Popen(
        ["python", "-c", formatted_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # get output and errors from the process
    out, err = res.communicate()
    out = out.replace("\n", "")

    return out, err


def _test_import(import_command, module):
    out, err = run_import_command(import_command, module)
    assert out == module, err


def test_import_torch():
    # arrange
    code = """
    import {MODULE}
    print({MODULE}.__name__)
    """
    module = "torch"

    # act & assert
    _test_import(code, module)


def test_import_nnmd():
    # arrange
    code = """
    import {MODULE}
    print({MODULE}.__name__)
    """
    module = "nnmd"

    # act & assert
    _test_import(code, module)
