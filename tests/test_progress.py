import io

import numpy as np
import pytest
from numba import njit, prange

from numba_progress import ProgressBar, ProgressBarType, __version__


# ---- Helpers (numba-compiled) ----

@njit(nogil=True)
def _numba_sequential(progress, n):
    for i in range(n):
        progress.update(1)


@njit(nogil=True)
def _numba_set(progress, val):
    progress.set(val)


@njit(nogil=True, parallel=True)
def _numba_parallel(progress, n):
    for i in prange(n):
        progress.update(1)


@njit(nogil=True)
def _numba_multi(bars, n):
    for i in range(n):
        bars[0].update(2)
        bars[1].update(1)


@njit(nogil=True)
def _numba_signature(n, progress):
    for i in range(n):
        progress.update(1)


# ---- Version ----

def test_version_exists():
    assert __version__ is not None


# ---- Python-level ProgressBar (no numba) ----

class TestPythonProgressBar:

    def test_initial_state(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        assert p.n == 0
        assert p.hook[0] == 0
        assert p.hook.dtype == np.uint64
        p.close()

    def test_update_increments(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        p.update(1)
        assert p.hook[0] == 1
        p.update(3)
        assert p.hook[0] == 4
        p.close()

    def test_set_overwrites(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        p.update(5)
        p.set(2)
        assert p.hook[0] == 2
        p.close()

    def test_context_manager_closes(self):
        with ProgressBar(total=10, file=io.StringIO()) as p:
            p.update(7)
        assert p.hook[0] == 7

    def test_update_zero_is_noop(self):
        buf = io.StringIO()
        with ProgressBar(total=10, file=buf) as p:
            p.update(0)
        assert p.hook[0] == 0

    def test_update_beyond_total(self):
        buf = io.StringIO()
        with ProgressBar(total=10, file=buf) as p:
            p.update(20)
        assert p.hook[0] == 20

    def test_bulk_update(self):
        buf = io.StringIO()
        with ProgressBar(total=100, file=buf) as p:
            p.update(100)
        assert p.hook[0] == 100

    def test_large_total(self):
        buf = io.StringIO()
        n = 10_000
        with ProgressBar(total=n, file=buf) as p:
            for i in range(n):
                p.update(1)
        assert p.hook[0] == n

    def test_custom_update_interval(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf, update_interval=0.5)
        assert p.update_interval == 0.5
        p.close()


# ---- Numba integration ----

class TestNumbaIntegration:

    def test_sequential_update(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        _numba_sequential(p, 10)
        p.close()
        assert p.hook[0] == 10

    def test_set_from_numba(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        p.update(8)
        _numba_set(p, 3)
        p.close()
        assert p.hook[0] == 3

    def test_parallel_update(self):
        buf = io.StringIO()
        n = 100
        p = ProgressBar(total=n, file=buf)
        _numba_parallel(p, n)
        p.close()
        assert p.hook[0] == n

    def test_explicit_signature(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        _numba_signature(10, p)
        p.close()
        assert p.hook[0] == 10

    def test_multiple_bars_tuple(self):
        buf1 = io.StringIO()
        buf2 = io.StringIO()
        p1 = ProgressBar(total=20, file=buf1)
        p2 = ProgressBar(total=40, file=buf2)
        _numba_multi((p1, p2), 20)
        p1.close()
        p2.close()
        assert p1.hook[0] == 40
        assert p2.hook[0] == 20

    def test_hook_still_writable_after_close(self):
        """The numpy hook array remains valid even after close (no reset guard)."""
        buf = io.StringIO()
        p = ProgressBar(total=5, file=buf)
        p.update(3)
        p.close()
        p.update(2)
        assert p.hook[0] == 5


# ---- Tqdm output correctness ----

class TestTqdmOutput:

    def test_final_output_contains_100_percent(self):
        buf = io.StringIO()
        with ProgressBar(total=5, file=buf) as p:
            for i in range(5):
                p.update(1)
        assert "100%" in buf.getvalue()

    def test_final_output_contains_total(self):
        buf = io.StringIO()
        with ProgressBar(total=5, file=buf) as p:
            for i in range(5):
                p.update(1)
        assert "5/5" in buf.getvalue()

    def test_partial_progress_shown(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        p.update(3)
        p.close()
        assert "30%" in buf.getvalue().replace(" ", "")

    def test_output_from_numba_sequential(self):
        buf = io.StringIO()
        p = ProgressBar(total=10, file=buf)
        _numba_sequential(p, 10)
        p.close()
        assert "100%" in buf.getvalue()

    def test_output_from_numba_parallel(self):
        buf = io.StringIO()
        n = 50
        p = ProgressBar(total=n, file=buf)
        _numba_parallel(p, n)
        p.close()
        assert "100%" in buf.getvalue()
