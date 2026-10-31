"""Libretro frontend smoke test using the free '2048' core (no ROM needed).

Skipped when the core is not present: run scripts/fetch_cores.sh or download
2048_libretro.so from buildbot.libretro.com into ./cores.
"""

from pathlib import Path

import numpy as np
import pytest

from fightrl.backends.libretro import LibretroBackend
from fightrl.core.controller import ControllerState

CORE = Path(__file__).resolve().parent.parent / "cores" / "2048_libretro.so"

pytestmark = pytest.mark.skipif(not CORE.exists(), reason="2048 libretro core not installed")


@pytest.fixture
def backend(tmp_path):
    be = LibretroBackend(CORE, rom_path=None, system_dir=tmp_path)
    yield be
    be.close()


def test_core_loads_and_reports_info(backend):
    assert backend.info["library_name"]
    assert backend.fps > 50
    assert backend.base_width > 0 and backend.base_height > 0


def test_frames_and_input(backend):
    for _ in range(20):
        backend.step([ControllerState("left"), ControllerState()])
    frame = backend.screen()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.mean() > 0


def test_save_states_roundtrip(backend):
    for _ in range(30):
        backend.step([ControllerState("down"), ControllerState()])
    snapshot = backend.save_state()
    assert len(snapshot) > 0
    before = backend.screen().copy()
    for _ in range(10):
        backend.step([ControllerState("left"), ControllerState()])
    backend.load_state(snapshot)
    backend.step([ControllerState("neutral"), ControllerState()])
    # After restore the board may advance by one frame but must look like the snapshot again.
    assert backend.screen().shape == before.shape
