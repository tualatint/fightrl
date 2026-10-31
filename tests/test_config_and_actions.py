"""Tests for game configs, macros and the action encoding."""

import json
from pathlib import Path

import pytest

from fightrl.core.actions import Macro, MacroStep
from fightrl.core.config import PACKAGE_GAMES_DIR, GameConfig, list_games, load_game_config
from fightrl.core.controller import ControllerState, DIRECTIONS, mirror_direction, resolve_direction


def test_all_bundled_configs_load():
    games = list_games()
    assert {"mock", "sf2_snes", "sf2_genesis", "samsho2_neogeo"} <= {g["id"] for g in games}
    for game in games:
        cfg = load_game_config(game["id"])
        assert cfg.id == game["id"]
        assert cfg.action_buttons
        assert cfg.macros
        if cfg.backend == "libretro":
            assert cfg.core and cfg.rom
            for button in cfg.action_buttons:
                assert button in cfg.button_map, f"{game['id']}: {button} missing from button_map"


def test_macro_action_space_matches_config():
    cfg = load_game_config("sf2_snes")
    names = [m.name for m in cfg.macros]
    assert "walk_fwd" in names and "attack_HP" in names and "hadoken_punch" in names
    total = sum(m.total_frames for m in cfg.macros)
    assert total > 0


def test_special_macro_facing_resolution():
    cfg = load_game_config("sf2_snes")
    hadoken = next(m for m in cfg.macros if m.name == "hadoken_punch")
    facing_right = [s.controller(1) for s in hadoken.steps]
    facing_left = [s.controller(-1) for s in hadoken.steps]
    assert facing_right[2].direction == "right"
    assert facing_left[2].direction == "left"
    assert facing_right[0].buttons == frozenset()


def test_mirror_and_resolve():
    assert mirror_direction("up_left") == "up_right"
    assert mirror_direction("down_right") == "down_left"
    assert mirror_direction("neutral") == "neutral"
    assert resolve_direction("back_down", 1) == "down_left"
    assert resolve_direction("back_up", -1) == "up_right"
    with pytest.raises(ValueError):
        resolve_direction("sideways", 1)


def test_macro_dict_roundtrip():
    macro = Macro("test", [MacroStep("down", ("HP",), 3), MacroStep("fwd", (), 2)])
    assert Macro.from_dict(macro.to_dict()) == macro
    assert macro.total_frames == 5
    assert macro.steps[0].controller(1) == ControllerState("down", frozenset({"HP"}))


def test_real_games_default_to_hybrid_skills():
    for game in ("sf2_snes", "sf2_genesis", "samsho2_neogeo"):
        cfg = load_game_config(game)
        assert cfg.action_mode_default == "hybrid", game
        assert cfg.n_primitives == len(DIRECTIONS) * (1 + len(cfg.action_buttons))
    names = [m.name for m in load_game_config("samsho2_neogeo").macros]
    assert {"qcf_A", "dp_A", "qcb_A"} <= set(names)


def test_hybrid_skills_only_use_configured_buttons():
    for game in list_games():
        cfg = load_game_config(game["id"])
        for macro in cfg.macros:
            for step in macro.steps:
                assert set(step.buttons) <= set(cfg.action_buttons), (game["id"], macro.name)


def test_generic_skills_press_a_button_the_game_has():
    for game in ("mock", "samsho2_neogeo"):
        cfg = load_game_config(game)
        for name in ("jump_attack", "crouch_attack"):
            macro = next(m for m in cfg.macros if m.name == name)
            pressed = {b for step in macro.steps for b in step.buttons}
            assert pressed and pressed <= set(cfg.action_buttons), (game, name)


def test_primitive_buttons_must_exist():
    with pytest.raises(ValueError, match="unknown buttons"):
        GameConfig.from_dict({
            "id": "x",
            "action_buttons": ["LP"],
            "primitive_buttons": [["LP", "LK"]],
        })


def test_action_mode_is_validated():
    assert GameConfig.from_dict({"id": "x", "action_mode": "hybrid"}).action_mode_default == "hybrid"
    with pytest.raises(ValueError, match="unknown action_mode"):
        GameConfig.from_dict({"id": "x", "action_mode": "combo"})
