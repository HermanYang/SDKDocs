import os
import tempfile

TMP_DIR = os.getenv("TMPDIR") or ("/fast/algo/tmp"
                                  if os.path.exists("/fast/algo/tmp") else "/tmp")


def check_kwargs(**kwargs):
    if "dir" in kwargs:
        raise ValueError("Cannot set dir in keyword arguments")


def mkdtemp(**kwargs):
    check_kwargs(**kwargs)
    return tempfile.mkdtemp(dir=TMP_DIR, **kwargs)


def named_temp_file(**kwargs):
    check_kwargs(**kwargs)
    return tempfile.NamedTemporaryFile(dir=TMP_DIR, **kwargs)
