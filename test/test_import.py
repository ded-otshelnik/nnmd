#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import inspect
import subprocess

def python_code_as_command(python_code, module_name):
    """Refactor python code for running as terminal commands (python -c "<code>"),
    not like file .py

    Args:
        python_code: code that must be formatted
        module_name: module that must be imported
    """
    python_code = inspect.cleandoc(python_code) \
                         .format(MODULE=module_name)
    return python_code.replace("\n", ";")

def run_command(code: list, module: str):
    """Run python code in terminal
    """
    formatted_code = python_code_as_command(code, module)

    # run code in new process
    res = subprocess.Popen(["python", "-c", formatted_code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # get out and errors from the process
    out, err = res.communicate()
    out = out.replace('\n', '')

    return out, err


def test_import_nnmd():
    # arrange
    code = """
    import torch
    import {MODULE}
    print({MODULE}.__name__)
    """
    module = "nnmd"

    # act
    out, err = run_command(code, module)

    # assert
    assert out == module, err

def test_import_nnmd_cpp():
    # arrange
    code = """
    import torch
    from nnmd import {MODULE}
    print({MODULE}.__name__)
    """
    module = "nnmd_cpp"

    # Act
    out, err = run_command(code, module)

    # Assert
    assert out == module, err