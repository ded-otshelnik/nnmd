#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import subprocess
TORCH_EXTENSION_NAME = 'mdnn_cpp'

def python_code_as_command(python_code):
    python_code = inspect.cleandoc(python_code)
    return python_code.replace("TORCH_EXTENSION_NAME", TORCH_EXTENSION_NAME).replace("\n", ";")


def test_import():
    code = """
    import torch
    import {TORCH_EXTENSION_NAME}
    
    print({TORCH_EXTENSION_NAME}.__name__)
    """
    res = subprocess.Popen(
        ["python", "-c", python_code_as_command(code)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(res)
    assert res == f"{TORCH_EXTENSION_NAME}"

test_import()