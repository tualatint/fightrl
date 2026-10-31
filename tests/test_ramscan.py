"""The RAM scanner must rediscover the mock game's health/timer addresses."""

from fightrl.backends.mock import ADDR_P1_HP, ADDR_P2_HP, ADDR_TIMER, MockFighter
from fightrl.core.state import VisionStateReader
from fightrl.ramscan import BrawlerBot, RamScanner


def test_scanner_finds_big_endian_health():
    """Neo Geo stores 16-bit counters high-byte-first."""
    scanner = RamScanner(64)
    values = [100, 100, 80, 80, 40, 40, 0, 0]
    ref = []
    for i, value in enumerate(values):
        raw = bytearray(64)
        raw[10] = value >> 8
        raw[11] = value & 0xFF
        scanner.update(bytes(raw))
        if i in (2, 4, 6):
            ref.append(i)
    candidates = scanner.health_candidates(ref)
    word = [c for c in candidates if c["addr"] == 10 and c["width"] == 2]
    assert word and word[0]["endian"] == "big"
    assert word[0]["init"] == 100 and word[0]["min"] == 0


def test_scanner_finds_health_and_timer():
    backend = MockFighter()
    backend.reset()
    vision = VisionStateReader()
    assert vision.calibrate(backend.screen())
    scanner = RamScanner(backend.ram_size)
    p1_bot, p2_bot = BrawlerBot(1, ["LP", "HP", "LK", "HK"], seed=5), BrawlerBot(2, ["LP", "HP", "LK", "HK"], seed=5)
    prev = vision.read(backend)
    ref_p1, ref_p2 = [], []
    for _ in range(900):
        scanner.update(backend.read_ram(0, backend.ram_size))
        state = vision.read(backend)
        if prev.valid and state.valid:
            if state.p1_frac < prev.p1_frac - 0.01:
                ref_p1.append(scanner.frame)
            if state.p2_frac < prev.p2_frac - 0.01:
                ref_p2.append(scanner.frame)
        prev = state
        backend.step([p1_bot.act(None, {}, state), p2_bot.act(None, {}, state)])

    assert len(ref_p1) > 3 and len(ref_p2) > 3
    p1 = scanner.health_candidates(ref_p1, top=3)
    p2 = scanner.health_candidates(ref_p2, top=3)
    assert p1 and p1[0]["addr"] == ADDR_P1_HP and p1[0]["width"] == 1
    assert p2 and p2[0]["addr"] == ADDR_P2_HP and p2[0]["width"] == 1
    assert p1[0]["init"] == 176

    timers = scanner.timer_candidates(top=3)
    assert timers and timers[0]["addr"] == ADDR_TIMER
    assert timers[0]["init"] == 99
