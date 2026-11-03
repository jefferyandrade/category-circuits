"""Write-once output files and folders for the analyses.

A file is written to a hidden sibling, flushed to disk and hard-linked into place, so it appears whole or not at all
and never replaces an existing file. Several files are published together or not at all. A result folder is built in
a temporary folder and its files are linked into a new folder. Outputs that overlap an input, or each other, are
refused. Readers should not read a destination while it is being published.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
import shutil
import uuid
from pathlib import Path


def validate_paths(outputs, inputs=()):
    """Refuse outputs that are the same path as, or lie inside, another output or an input."""
    outs = [Path(os.path.realpath(os.fspath(p))) for p in outputs if p is not None]
    ins = [Path(os.path.realpath(os.fspath(p))) for p in inputs if p is not None]
    for i, out in enumerate(outs):
        for other in outs[:i]:
            same_file = out.exists() and other.exists() and os.path.samefile(out, other)
            if same_file or out == other or out in other.parents or other in out.parents:
                raise ValueError(f"output destinations overlap: {out}, {other}")
        for inp in ins:
            same_file = out.exists() and inp.exists() and os.path.samefile(out, inp)
            if same_file or out == inp or out in inp.parents or (inp.is_dir() and inp in out.parents):
                raise ValueError(f"output overlaps an input: {out}, {inp}")
    return outs


def check_outputs(files=(), dirs=(), inputs=()):
    """Check every destination before reading inputs or calculating a result."""
    files = [Path(p) for p in files if p is not None]
    dirs = [Path(p) for p in dirs if p is not None]
    for path in files:
        if os.path.lexists(path):
            raise FileExistsError("output already exists: %s" % path)
    for path in dirs:
        if os.path.lexists(path) and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError("output folder is not empty: %s" % path)
    try:
        validate_paths(files + dirs, inputs)
    except ValueError as error:
        raise SystemExit(str(error))


def prepare_dir(dest, inputs=()):
    """Reserve a new or empty output folder before doing any work."""
    check_outputs(dirs=[dest], inputs=inputs)
    Path(dest).mkdir(parents=True, exist_ok=True)


@contextmanager
def staged_file(dest):
    """Write at a temporary sibling path, then publish the finished file exclusively."""
    dest = Path(dest)
    check_outputs(files=[dest])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (".%s.partial-%s%s" % (dest.stem, uuid.uuid4().hex, dest.suffix))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    os.close(fd)
    try:
        yield tmp
        publish_file(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def write_file(dest, mode="w", **kwargs):
    """Open a temporary sibling for text or binary output and publish it when closed."""
    with staged_file(dest) as tmp:
        with tmp.open(mode, **kwargs) as stream:
            yield stream


def _stage(value, dest):
    dest = Path(dest)
    if isinstance(value, Path):
        if value.parent.resolve() != dest.parent.resolve():
            raise ValueError("temporary file must be beside its output: %s" % dest)
        with value.open("rb") as stream:
            os.fsync(stream.fileno())
        return value
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (".%s.partial-%s" % (dest.name, uuid.uuid4().hex))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as fh:
            if hasattr(value, "to_csv"):
                value = value.to_csv(index=False).encode("utf-8")
            elif isinstance(value, str):
                value = value.encode("utf-8")
            fh.write(value)
            fh.flush()
            os.fsync(fh.fileno())
        return tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_files(products):
    """Write every (value, destination) pair, or none of them.

    A value is a table (written as CSV), text or bytes. If writing or linking any file fails, the files this call
    linked are removed again. An existing destination is never replaced.
    """
    products = list(products)
    for _, dest in products:
        if os.path.lexists(dest):
            raise FileExistsError(dest)
    dests = validate_paths([dest for _, dest in products])
    staged, linked = [], []
    try:
        for dest in dests:
            if os.path.lexists(dest):
                raise FileExistsError(dest)
        for (value, _), dest in zip(products, dests):
            staged.append((_stage(value, dest), dest))
        for tmp, dest in staged:
            os.link(tmp, dest)
            linked.append(dest)
        for parent in {d.parent for d in dests}:
            _sync_dir(parent)
    except BaseException:
        for dest in linked:
            dest.unlink(missing_ok=True)
        raise
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)


def publish_file(df_or_bytes, dest):
    """Write a table, text, bytes or a completed temporary sibling Path to a new file."""
    publish_files([(df_or_bytes, dest)])


def publish_dir(tmp_dir, dest):
    """Move the files of a finished temporary folder into the new folder dest.

    dest must be new or empty. meta.json is linked last, so a folder without it is
    unfinished. If linking fails, dest is removed again; on success the temporary folder is removed.
    """
    tmp_dir = Path(tmp_dir)
    check_outputs(dirs=[dest], inputs=[tmp_dir])
    dest = validate_paths([dest], [tmp_dir])[0]
    if dest.exists():
        dest.rmdir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(dest)
    try:
        sources = sorted(tmp_dir.rglob("*"))
        sources.sort(key=lambda p: (p.name == "meta.json", str(p)))
        for src in sources:
            rel = src.relative_to(tmp_dir)
            target = dest / rel
            if src.is_dir():
                target.mkdir()
            else:
                with src.open("rb") as fh:
                    os.fsync(fh.fileno())
                os.link(src, target)
        for folder in sorted((p for p in dest.rglob("*") if p.is_dir()),
                             key=lambda p: len(p.parts), reverse=True):
            _sync_dir(folder)
        _sync_dir(dest)
        _sync_dir(dest.parent)
    except BaseException:
        shutil.rmtree(dest)
        raise
    else:
        shutil.rmtree(tmp_dir)
