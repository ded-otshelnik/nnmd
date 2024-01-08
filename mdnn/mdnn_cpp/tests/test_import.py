#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import subprocess

def python_code_as_command(python_code):
    python_code = inspect.cleandoc(python_code).format(TORCH_EXTENSION_NAME="mdnn_cpp")
    return python_code.replace("\n", ";")

def test_import():
    code = """
    import torch
    import {TORCH_EXTENSION_NAME}
    print({TORCH_EXTENSION_NAME}.__name__)
    """
    res = subprocess.Popen(
        ["python", "-c", python_code_as_command(code)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = res.communicate()
    out = out.replace('\n', '')
    assert out == "mdnn_cpp"

test_import()