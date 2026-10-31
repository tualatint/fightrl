"""Minimal headless libretro frontend (ctypes).

Runs any libretro core (snes9x, genesis_plus_gx, fbneo, ...) without a
window: video frames, joypad input, save states and system RAM are all
accessible, which is everything an RL environment needs.

The implementation is deliberately dependency free -- no RetroArch install
is required, only the core ``.so`` and the ROM you own.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.controller import ControllerState
from .base import Backend

# --------------------------------------------------------------------------- #
# libretro ABI
# --------------------------------------------------------------------------- #
RETRO_DEVICE_JOYPAD = 1
RETRO_DEVICE_ANALOG = 5
RETRO_DEVICE_ID_JOYPAD_MASK = 256

PIXEL_FORMAT_0RGB1555 = 0
PIXEL_FORMAT_XRGB8888 = 1
PIXEL_FORMAT_RGB565 = 2

MEMORY_SYSTEM_RAM = 2

ENV_SET_PIXEL_FORMAT = 10
ENV_SET_INPUT_DESCRIPTORS = 11
ENV_GET_CAN_DUPE = 3
ENV_SET_MESSAGE = 6
ENV_GET_SYSTEM_DIRECTORY = 9
ENV_GET_VARIABLE = 15
ENV_SET_VARIABLES = 16
ENV_GET_VARIABLE_UPDATE = 17
ENV_SET_SUPPORT_NO_GAME = 18
ENV_GET_LOG_INTERFACE = 27
ENV_GET_LIBRETRO_PATH = 19
ENV_SET_CONTROLLER_INFO = 35
ENV_SET_GEOMETRY = 37
ENV_GET_LANGUAGE = 39
ENV_GET_CORE_ASSETS_DIRECTORY = 30
ENV_GET_SAVE_DIRECTORY = 31
ENV_GET_AUDIO_VIDEO_ENABLE = 47
ENV_GET_FASTFORWARDING = 49
ENV_GET_INPUT_BITMASKS = 52
ENV_GET_PREFERRED_HW_RENDER = 56

_DIR_TO_BUTTON = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}


class retro_game_info(C.Structure):
    _fields_ = [
        ("path", C.c_char_p),
        ("data", C.c_void_p),
        ("size", C.c_size_t),
        ("meta", C.c_char_p),
    ]


class retro_system_info(C.Structure):
    _fields_ = [
        ("library_name", C.c_char_p),
        ("library_version", C.c_char_p),
        ("valid_extensions", C.c_char_p),
        ("need_fullpath", C.c_bool),
        ("block_extract", C.c_bool),
    ]


class retro_game_geometry(C.Structure):
    _fields_ = [
        ("base_width", C.c_uint),
        ("base_height", C.c_uint),
        ("max_width", C.c_uint),
        ("max_height", C.c_uint),
        ("aspect_ratio", C.c_float),
    ]


class retro_system_timing(C.Structure):
    _fields_ = [("fps", C.c_double), ("sample_rate", C.c_double)]


class retro_system_av_info(C.Structure):
    _fields_ = [("geometry", retro_game_geometry), ("timing", retro_system_timing)]


class retro_variable(C.Structure):
    _fields_ = [("key", C.c_char_p), ("value", C.c_char_p)]


_ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
_VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
_AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
_AUDIO_BATCH_CB = C.CFUNCTYPE(C.c_size_t, C.POINTER(C.c_int16), C.c_size_t)
_INPUT_POLL_CB = C.CFUNCTYPE(None)
_INPUT_STATE_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)

# Cores keep raw pointers to these callbacks; keep a module-level reference.
_ALIVE: list[Any] = []


class LibretroCore:
    """Thin wrapper around a loaded libretro core library."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.lib = C.CDLL(self.path)
        self._bind()

    def _bind(self) -> None:
        lib = self.lib
        lib.retro_set_environment.argtypes = [_ENV_CB]
        lib.retro_set_video_refresh.argtypes = [_VIDEO_CB]
        lib.retro_set_audio_sample.argtypes = [_AUDIO_CB]
        lib.retro_set_audio_sample_batch.argtypes = [_AUDIO_BATCH_CB]
        lib.retro_set_input_poll.argtypes = [_INPUT_POLL_CB]
        lib.retro_set_input_state.argtypes = [_INPUT_STATE_CB]
        lib.retro_init.argtypes = []
        lib.retro_deinit.argtypes = []
        lib.retro_api_version.restype = C.c_uint
        lib.retro_get_system_info.argtypes = [C.POINTER(retro_system_info)]
        lib.retro_get_system_av_info.argtypes = [C.POINTER(retro_system_av_info)]
        lib.retro_set_controller_port_device.argtypes = [C.c_uint, C.c_uint]
        lib.retro_reset.argtypes = []
        lib.retro_run.argtypes = []
        lib.retro_serialize_size.restype = C.c_size_t
        lib.retro_serialize.argtypes = [C.c_void_p, C.c_size_t]
        lib.retro_serialize.restype = C.c_bool
        lib.retro_unserialize.argtypes = [C.c_void_p, C.c_size_t]
        lib.retro_unserialize.restype = C.c_bool
        lib.retro_load_game.argtypes = [C.POINTER(retro_game_info)]
        lib.retro_load_game.restype = C.c_bool
        lib.retro_unload_game.argtypes = []
        lib.retro_get_region.restype = C.c_uint
        lib.retro_get_memory_data.argtypes = [C.c_uint]
        lib.retro_get_memory_data.restype = C.c_void_p
        lib.retro_get_memory_size.argtypes = [C.c_uint]
        lib.retro_get_memory_size.restype = C.c_size_t

    def system_info(self) -> dict[str, Any]:
        info = retro_system_info()
        self.lib.retro_get_system_info(C.byref(info))
        return {
            "library_name": (info.library_name or b"").decode("utf-8", "replace"),
            "library_version": (info.library_version or b"").decode("utf-8", "replace"),
            "valid_extensions": (info.valid_extensions or b"").decode("utf-8", "replace"),
            "need_fullpath": bool(info.need_fullpath),
        }


class LibretroBackend(Backend):
    """Two-player libretro game backend."""

    name = "libretro"
    has_ram = True

    def __init__(
        self,
        core_path: str | Path,
        rom_path: str | Path | None = None,
        system_dir: str | Path | None = None,
        button_map: Mapping[str, int] | None = None,
        core_options: Mapping[str, str] | None = None,
        capture_video: bool = True,
        n_ports: int = 2,
    ):
        self.core = LibretroCore(core_path)
        self.rom_path = str(rom_path) if rom_path else None
        self.system_dir = str(system_dir or Path(core_path).parent)
        Path(self.system_dir).mkdir(parents=True, exist_ok=True)
        self.button_map = dict(button_map or {})
        self.core_options = dict(core_options or {})
        self.capture_video = capture_video
        self.n_ports = n_ports

        self.pixel_format = PIXEL_FORMAT_0RGB1555
        self._raw_video: bytes | None = None
        self._video_dims: tuple[int, int, int] | None = None  # w, h, pitch
        self._frame: np.ndarray | None = None
        self._frame_dirty = True
        self._input_bits = [0, 0]
        self._system_dir_c = C.c_char_p(self.system_dir.encode())
        self._core_options_c: dict[str, bytes] = {}
        self._rom_info: retro_game_info | None = None
        self._closed = False

        self._env_cb = _ENV_CB(self._environment)
        self._video_cb = _VIDEO_CB(self._video_refresh)
        self._audio_cb = _AUDIO_CB(self._audio_sample)
        self._audio_batch_cb = _AUDIO_BATCH_CB(self._audio_batch)
        self._input_poll_cb = _INPUT_POLL_CB(self._input_poll)
        self._input_state_cb = _INPUT_STATE_CB(self._input_state)
        _ALIVE.extend([self._env_cb, self._video_cb, self._audio_cb,
                       self._audio_batch_cb, self._input_poll_cb, self._input_state_cb])

        lib = self.core.lib
        lib.retro_set_environment(self._env_cb)
        lib.retro_set_video_refresh(self._video_cb)
        lib.retro_set_audio_sample(self._audio_cb)
        lib.retro_set_audio_sample_batch(self._audio_batch_cb)
        lib.retro_set_input_poll(self._input_poll_cb)
        lib.retro_set_input_state(self._input_state_cb)
        lib.retro_init()

        self.info = self.core.system_info()
        self._load_game()
        av = retro_system_av_info()
        lib.retro_get_system_av_info(C.byref(av))
        self.fps = float(av.timing.fps) or 60.0
        self.sample_rate = float(av.timing.sample_rate)
        self.base_width = int(av.geometry.base_width)
        self.base_height = int(av.geometry.base_height)
        for port in range(self.n_ports):
            lib.retro_set_controller_port_device(port, RETRO_DEVICE_JOYPAD)
        self._ram_ptr = lib.retro_get_memory_data(MEMORY_SYSTEM_RAM)
        self._ram_size = int(lib.retro_get_memory_size(MEMORY_SYSTEM_RAM))
        lib.retro_reset()

    # -- loading ------------------------------------------------------- #
    def _load_game(self) -> None:
        lib = self.core.lib
        if self.rom_path:
            if not os.path.exists(self.rom_path):
                raise FileNotFoundError(f"ROM not found: {self.rom_path}")
            rom_b = self.rom_path.encode()
            self._rom_info = retro_game_info(path=rom_b, data=None, size=0, meta=None)
            ok = lib.retro_load_game(C.byref(self._rom_info))
        else:
            ok = lib.retro_load_game(None)
        if not ok:
            raise RuntimeError(
                f"core {self.info['library_name']} refused to load {self.rom_path!r}. "
                "Check that the ROM matches the core (e.g. sf2turbo.sfc for snes9x, "
                "samsho2.zip + neogeo.zip for fbneo)."
            )

    # -- libretro callbacks -------------------------------------------- #
    def _environment(self, cmd: int, data: Any) -> bool:
        try:
            if cmd == ENV_GET_SYSTEM_DIRECTORY or cmd == ENV_GET_CORE_ASSETS_DIRECTORY:
                C.cast(data, C.POINTER(C.c_char_p))[0] = self._system_dir_c
                return True
            if cmd == ENV_GET_SAVE_DIRECTORY:
                C.cast(data, C.POINTER(C.c_char_p))[0] = self._system_dir_c
                return True
            if cmd == ENV_SET_PIXEL_FORMAT:
                self.pixel_format = C.cast(data, C.POINTER(C.c_uint))[0]
                return True
            if cmd == ENV_GET_CAN_DUPE:
                C.cast(data, C.POINTER(C.c_bool))[0] = True
                return True
            if cmd == ENV_GET_INPUT_BITMASKS:
                C.cast(data, C.POINTER(C.c_bool))[0] = True
                return True
            if cmd == ENV_GET_VARIABLE:
                var = C.cast(data, C.POINTER(retro_variable))[0]
                key = (var.key or b"").decode("utf-8", "replace")
                value = self.core_options.get(key)
                if value is None:
                    return False
                encoded = value.encode()
                self._core_options_c[key] = encoded
                var.value = encoded
                return True
            if cmd == ENV_GET_VARIABLE_UPDATE:
                C.cast(data, C.POINTER(C.c_bool))[0] = False
                return True
            if cmd == ENV_GET_LANGUAGE:
                C.cast(data, C.POINTER(C.c_uint))[0] = 0  # English
                return True
            if cmd == ENV_GET_AUDIO_VIDEO_ENABLE:
                C.cast(data, C.POINTER(C.c_int))[0] = 3  # audio+video enabled
                return True
            if cmd == ENV_GET_FASTFORWARDING:
                C.cast(data, C.POINTER(C.c_bool))[0] = False
                return True
            if cmd == ENV_GET_LIBRETRO_PATH:
                C.cast(data, C.POINTER(C.c_char_p))[0] = C.c_char_p(self.core.path.encode())
                return True
            return False
        except Exception:
            return False

    def _video_refresh(self, data: int, width: int, height: int, pitch: int) -> None:
        if not data:
            return  # duplicate frame, keep the previous buffer
        self._video_dims = (int(width), int(height), int(pitch))
        if not self.capture_video:
            return
        bpp = 4 if self.pixel_format == PIXEL_FORMAT_XRGB8888 else 2
        n = int(pitch) * int(height)
        self._raw_video = C.string_at(data, n)
        self._frame_dirty = True

    def _audio_sample(self, left: int, right: int) -> None:
        pass

    def _audio_batch(self, data: Any, frames: int) -> int:
        return frames

    def _input_poll(self) -> None:
        pass

    def _input_state(self, port: int, device: int, index: int, id: int) -> int:
        if device != RETRO_DEVICE_JOYPAD:
            return 0
        bits = self._input_bits[port] if port < len(self._input_bits) else 0
        if id == RETRO_DEVICE_ID_JOYPAD_MASK:
            return bits
        if 0 <= id < 16:
            return 1 if (bits >> id) & 1 else 0
        return 0

    # -- Backend API ---------------------------------------------------- #
    def _encode(self, controller: ControllerState) -> int:
        bits = 0
        dx, dy = controller.vector()
        if dy < 0:
            bits |= 1 << self.button_map.get("UP", 4)
        if dy > 0:
            bits |= 1 << self.button_map.get("DOWN", 5)
        if dx < 0:
            bits |= 1 << self.button_map.get("LEFT", 6)
        if dx > 0:
            bits |= 1 << self.button_map.get("RIGHT", 7)
        for button in controller.buttons:
            bit = self.button_map.get(button)
            if bit is not None:
                bits |= 1 << int(bit)
        return bits

    def step(self, controllers: Sequence[ControllerState]) -> None:
        for port in range(min(self.n_ports, len(self._input_bits))):
            ctrl = controllers[port] if port < len(controllers) else ControllerState()
            self._input_bits[port] = self._encode(ctrl)
        self.core.lib.retro_run()

    def reset(self, seed: int | None = None) -> None:
        self.core.lib.retro_reset()
        self._frame_dirty = True

    def screen(self) -> np.ndarray:
        if not self.capture_video:
            raise RuntimeError("video capture is disabled for this backend")
        if self._frame is not None and not self._frame_dirty:
            return self._frame
        if self._raw_video is None or self._video_dims is None:
            return np.zeros((self.base_height or 224, self.base_width or 320, 3), np.uint8)
        w, h, pitch = self._video_dims
        fmt = self.pixel_format
        if fmt == PIXEL_FORMAT_XRGB8888:
            arr = np.frombuffer(self._raw_video, np.uint8).reshape(h, pitch // 4, 4)[:, :w]
            frame = np.ascontiguousarray(arr[..., [2, 1, 0]])
        else:
            arr = np.frombuffer(self._raw_video, np.uint16).reshape(h, pitch // 2)[:, :w]
            if fmt == PIXEL_FORMAT_RGB565:
                r = ((arr >> 11) & 0x1F).astype(np.uint16) * 255 // 31
                g = ((arr >> 5) & 0x3F).astype(np.uint16) * 255 // 63
                b = (arr & 0x1F).astype(np.uint16) * 255 // 31
            else:  # 0RGB1555
                r = ((arr >> 10) & 0x1F).astype(np.uint16) * 255 // 31
                g = ((arr >> 5) & 0x1F).astype(np.uint16) * 255 // 31
                b = (arr & 0x1F).astype(np.uint16) * 255 // 31
            frame = np.ascontiguousarray(np.stack([r, g, b], axis=-1).astype(np.uint8))
        self._frame = frame
        self._frame_dirty = False
        return frame

    def read_ram(self, addr: int, size: int) -> bytes:
        if not self._ram_ptr or addr < 0 or addr + size > self._ram_size:
            raise IndexError(f"RAM read out of range: {addr:#x}+{size} (size {self._ram_size:#x})")
        return C.string_at(self._ram_ptr + addr, size)

    @property
    def ram_size(self) -> int:
        return self._ram_size

    def save_state(self) -> bytes:
        size = int(self.core.lib.retro_serialize_size())
        if size == 0:
            raise RuntimeError("core does not support save states")
        buf = C.create_string_buffer(size)
        if not self.core.lib.retro_serialize(buf, size):
            raise RuntimeError("retro_serialize failed")
        return buf.raw

    def load_state(self, data: bytes) -> None:
        buf = C.create_string_buffer(bytes(data), len(data))
        if not self.core.lib.retro_unserialize(buf, len(data)):
            raise RuntimeError("retro_unserialize failed (state from a different core/ROM?)")
        self._frame_dirty = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.core.lib.retro_unload_game()
        finally:
            self.core.lib.retro_deinit()


def load_state_file(path: str | Path) -> bytes:
    return Path(path).expanduser().read_bytes()


def save_state_file(path: str | Path, data: bytes) -> None:
    Path(path).expanduser().write_bytes(data)
