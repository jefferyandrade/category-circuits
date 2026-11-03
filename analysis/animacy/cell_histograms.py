"""Cell histograms of the label permutations of the shared-category animacy statistics, on one thread.

A small C++ loop counts each incidence signature's animate categories under a permutation and adds its
multiplicity; the Python side evaluates the float32 per-cell specificity. The loop is compiled once into
``scratch/native`` beside this file; without a compiler a NumPy implementation runs instead. The permutation draws
are the same either way.
"""
import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

SCRATCH = Path(__file__).resolve().parent / 'scratch'

SOURCE = r'''
#include <cstdint>
extern "C" void cells(const uint64_t* words, const int64_t* mult,
                      const int64_t* tot, const uint64_t* perms,
                      int64_t R, int NW, int NP, int K, int A, int64_t* out) {
  const int stride = (K+1)*(A+1);
  for(int p=0;p<NP;p++) {
    int64_t* h = out + int64_t(p)*stride;
    const uint64_t* pw = perms + p*NW;
    for(int64_t r=0;r<R;r++) {
      if(tot[r] == 0) continue;
      int a=0;
      for(int w=0;w<NW;w++) a += __builtin_popcountll(words[r*NW+w] & pw[w]);
      h[tot[r]*(A+1)+a] += mult[r];
    }
  }
}
extern "C" void ordered_sum(const uint64_t* words, const int64_t* mult,
                           const int64_t* tot, const uint64_t* perms,
                           int64_t R, int NW, int NP, int A,
                           const float* lookup, float* out) {
  for(int p=0;p<NP;p++) {
    float acc = 0;
    for(int64_t r=0;r<R;r++) {
      if(tot[r]==0) continue;
      int a=0;
      for(int w=0;w<NW;w++) a += __builtin_popcountll(words[r*NW+w] & perms[p*NW+w]);
      acc += float(mult[r]) * lookup[tot[r]*(A+1)+a];
    }
    out[p]=acc;
  }
}
'''
_DLL = None


def _compiler_environment():
    """The environment of the compiler, with its temporary and cache folders under SCRATCH."""
    env = dict(os.environ)
    for name, rel in [('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'), ('CLANG_MODULE_CACHE_PATH', 'clang'),
                      ('CCACHE_DIR', 'ccache')]:
        folder = SCRATCH / rel
        folder.mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    return env


def _dll():
    global _DLL
    if _DLL is not None:
        return _DLL
    directory = SCRATCH / 'native'
    directory.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(SOURCE.encode()).hexdigest()[:16]
    source = directory / (tag + '.cpp')
    binary = directory / (tag + ('.dylib' if sys.platform == 'darwin' else '.so'))
    if not binary.exists():
        source.write_text(SOURCE)
        cmd = ['c++', '-O3', '-std=c++17', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
               '-fPIC', str(source), '-o', str(binary)]
        subprocess.run(cmd, env=_compiler_environment(), check=True, capture_output=True)
    _DLL = ctypes.CDLL(str(binary))
    _DLL.cells.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    _DLL.cells.restype = None
    _DLL.ordered_sum.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    _DLL.ordered_sum.restype = None
    return _DLL


def cell_histograms(words, mult, total, permutations, K, A):
    words = np.ascontiguousarray(words, np.uint64)
    mult = np.ascontiguousarray(mult, np.int64)
    total = np.ascontiguousarray(total, np.int64)
    permutations = np.ascontiguousarray(permutations, np.uint64)
    out = np.zeros((len(permutations), K + 1, A + 1), np.int64)
    try:
        dll = _dll()
    except (OSError, subprocess.CalledProcessError):
        for j, p in enumerate(permutations):
            a = np.bitwise_count(words & p).sum(axis=1)
            use = total > 0
            out[j] = np.bincount(total[use] * (A + 1) + a[use].astype(np.int64),
                                 weights=mult[use], minlength=(K + 1) * (A + 1)).reshape(K + 1, A + 1)
        return out
    dll.cells(words.ctypes.data, mult.ctypes.data, total.ctypes.data, permutations.ctypes.data,
              len(words), words.shape[1], len(permutations), K, A, out.ctypes.data)
    return out


def ordered_sum(words, mult, total, permutations, A, lookup):
    words, permutations = np.ascontiguousarray(words, np.uint64), np.ascontiguousarray(permutations, np.uint64)
    mult, total = np.ascontiguousarray(mult, np.int64), np.ascontiguousarray(total, np.int64)
    lookup = np.ascontiguousarray(lookup, np.float32)
    out = np.zeros(len(permutations), np.float32)
    try:
        dll = _dll()
    except (OSError, subprocess.CalledProcessError):
        for p, flags in enumerate(permutations):
            a = np.bitwise_count(words & flags).sum(axis=1).astype(np.int64)
            values = mult.astype(np.float32) * lookup[total, a]
            out[p] = np.add.accumulate(values[total > 0], dtype=np.float32)[-1]
        return out
    dll.ordered_sum(words.ctypes.data, mult.ctypes.data, total.ctypes.data, permutations.ctypes.data,
                    len(words), words.shape[1], len(permutations), A, lookup.ctypes.data, out.ctypes.data)
    return out
