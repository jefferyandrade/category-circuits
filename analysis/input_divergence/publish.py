"""Write-once output folders: a finished folder is renamed into place without replacing an existing one (a rename that
refuses to replace where the system has one, otherwise an exclusive creation and a move of each file).
"""
from __future__ import annotations

import ctypes
import errno
import os
import sys

_NO_NATIVE_FLAG = {errno.EINVAL, errno.ENOSYS, getattr(errno, "ENOTSUP", errno.EINVAL),
                   getattr(errno, "EOPNOTSUPP", errno.EINVAL)}


def _rename_portable(source, target):
    try:
        os.mkdir(target)
    except FileExistsError as exc:
        raise FileExistsError(f"output already exists: {target}") from exc
    for name in os.listdir(source):
        os.rename(os.path.join(source, name), os.path.join(target, name))
    os.rmdir(source)


def rename_new(source, target):
    """Move the finished folder ``source`` to ``target``, refusing to replace an existing ``target``."""
    if not os.listdir(source):
        raise ValueError(f"refusing to publish an empty folder: {source}")
    lib = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(lib, "renamex_np"):
        fn = lib.renamex_np
        fn.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        code = fn(os.fsencode(source), os.fsencode(target), 4)
    elif hasattr(lib, "renameat2"):
        fn = lib.renameat2
        fn.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        code = fn(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    else:
        return _rename_portable(source, target)
    if code:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(f"output already exists: {target}")
        if error in _NO_NATIVE_FLAG:
            return _rename_portable(source, target)
        raise OSError(error, os.strerror(error), str(target))
