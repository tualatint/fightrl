# FightRL — reinforcement-learning environments for 2D fighting games

FightRL turns **Street Fighter II** (SNES / Genesis) and **Samurai Shodown II**
(Neo Geo) into headless Gymnasium environments and ships a complete training
stack: PPO, self-play league, scripted opponents, a RAM-address discovery tool,
and replay export.

Two backends are included:

| Backend | What it runs | Needs |
|---|---|---|
| `libretro` | real ROMs through any libretro core (`snes9x`, `genesis_plus_gx`, `fbneo`) | a core `.so` + **your own ROM dump** |
| `mock` | a small built-in 2D fighter (no copyrighted content) | nothing |

The mock game exists so the whole pipeline — observations, rewards, vision
health reading, RAM scanning, self-play, evaluation — can be developed, tested
and benchmarked without a ROM. It renders a Street-Fighter-style HUD, has
hitstun/blockstun/jumps/rounds, and exposes a game-RAM image just like a real
console would.

> **No ROMs or BIOS files are bundled or downloaded.** You must supply dumps
> you are legally allowed to use. Cores themselves are open source and are
> fetched by `scripts/fetch_cores.sh`.

## Architecture

```
              ┌──────────────────────────── FightEnv (Gymnasium) ───────────────────────────┐
              │                                                                              │
  action ───► │  action decoder        backend.step([p1, p2])        observation encoder     │──► obs
  (macro, raw │  macro / 8-way+6btn ──► libretro core  ──┐          crop → gray → 84×84 →    │   Dict(pixels,
   or hybrid) │                          or mock engine │          stack 4                  │        state)
              │                                          ▼                                     │
              │  reward  ◄────────────  GameState  ◄── state reader (RAM map | vision HUD)    │
              │  damage dealt/taken, round win, life-lead shaping                                │
              │                                                                              │
              │  opponent: scripted bot | frozen checkpoint | self-play league                  │
              └──────────────────────────────────────────────────────────────────────────────┘
```

## Install

```bash
pip install -e .            # numpy, gymnasium, stable-baselines3, torch, imageio
pip install -e ".[dev]"     # + pytest
pip install opencv-python-headless   # optional, ~2x faster frame preprocessing
```

Python 3.10+. A GPU is nice for training but not required.

## Quickstart (no ROM, 2 minutes)

```bash
export PYTHONPATH=.       # or pip install -e .

python -m fightrl.train --game mock --opponent aggressive \
    --steps 20000 --n-envs 4 --out runs/mock-demo

python -m fightrl.evaluate --game mock --model runs/mock-demo/best.zip \
    --opponent aggressive --episodes 10 --video runs/mock-demo/replay.mp4
```

Sanity checks:

```bash
python -m pytest tests -q
python -m fightrl.train --list-games
```

## Running a real game

### 1. Get the emulator cores

```bash
bash scripts/fetch_cores.sh          # snes9x, genesis_plus_gx, fbneo -> ./cores
```

### 2. Put your ROMs in place

```
roms/sf2turbo.sfc           # Street Fighter II Turbo (SNES)
roms/sf2sce.md              # Street Fighter II' SCE (Genesis)
roms/samsho2.zip            # Samurai Shodown II romset
cores/neogeo.zip            # Neo Geo BIOS (required by FBNeo, not included)
```

Edit `fightrl/games/*.json` if your filenames differ.

### 3. Record a start state (skips menus)

The games boot into attract mode / menus. Play to the first frame of a match
and press **S**:

```bash
python -m fightrl.capture --game sf2_snes
```

Save as `states/sf2_snes.state` (the default path the config expects). Every
episode then starts exactly at the start of that round. Press **Esc** to quit.

If you have a second display or want to watch, this tool is also how you
sanity-check the controls and the button mapping.

### 4. Get exact health values (recommended)

The environment can read the HUD health bars from pixels out of the box
(`state_source: "auto"`), but RAM values are exact and ~2x cheaper. Discover
them automatically — the tool plays the game with two brawler bots and matches
memory changes to visible health drops:

```bash
python -m fightrl.ramscan --game sf2_snes --seconds 45 --out states/sf2_ram.json
```

It prints something like:

```
P1 health candidates
  0x1a2f40 width=1 f1=1.00 recall=1.00 precision=1.00 init=176 min=39 drops=14
P2 health candidates
  0x1a2f41 width=1 f1=1.00 recall=1.00 precision=1.00 init=176 min=0 drops=21
Timer candidates
  0x1a2f20 width=1 init=99 interval=60.0±0.0 decrements=17
```

Paste the `ram_map` block it prints into `fightrl/games/sf2_snes.json`
(or pass `--ram-map` at runtime — see below). Addresses are relative to the
RAM region the core exposes, so they survive restarts.

Prefer vision-only? Just clear `ram_map`; `state_source: "auto"` falls back to
the health bars.

### 5. Train

```bash
python -m fightrl.train --game sf2_snes \
    --opponent league:runs/sf2/league \
    --steps 5000000 --n-envs 8 --device cuda \
    --out runs/sf2 --save-every 100000
```

The league starts as a mix of scripted styles (**aggressive**, **random**,
**defensive**, **passive**) and then gradually becomes "the agent's own past
selves": every `--save-every` steps a frozen copy is published to
`runs/sf2/league`, and each environment picks a random opponent from that
directory at the start of every episode. A quarter of the episodes keep using
a scripted style even once checkpoints exist — without that regularizer the
policy forgets how to chase a retreating opponent. To continue an interrupted
run, pass `--resume runs/sf2/final.zip --steps <additional steps>`; the step
budget is added on top of the checkpoint's counter.

### 6. Evaluate / record

```bash
python -m fightrl.evaluate --game sf2_snes --model runs/sf2/best.zip \
    --opponent aggressive --episodes 20 --video runs/sf2/replay.mp4
```

### Samurai Shodown II notes (FBNeo)

`fightrl/games/samsho2_neogeo.json` runs the arcade romset in AES mode:

```
roms/samsho2.zip        # your dump
cores/neogeo.zip        # Neo Geo BIOS, required by FBNeo (not bundled)
states/samsho2_neogeo.state
```

The shipped start state is a **2P VS** match (Haohmaru vs Haohmaru, timer at
99, "CHALLENGER" label on P2) captured as the round begins. To record your own,
boot the game, pick 2P VS and a character, and press **S** once both bars are
full: `python -m fightrl.capture --game samsho2_neogeo`.

State comes from the HUD bars (`state_source: "auto"` with no RAM map): a
`ramscan` run did not find addresses that track both health bars on this
romset, so vision is the authoritative source. The rectangles and the hue
window live in the game JSON:

```json
"hud": {
  "p1_bar": [16, 18, 128, 12],
  "p2_bar": [176, 18, 128, 12],
  "hue": [95, 150]
}
```

A longer probe (120 s of brawling, plus a full-health-to-KO differential and a
whole-RAM monotonic scan) still found no byte or 16-bit word that follows the
bars: the only candidates that start at full and end at zero are animation and
intro counters, and the arcade cheat addresses (`0x100A46`/`0x100A4A` P1/P2
energy, `0x100AC6` timer) are refreshed by the game every frame, so writing
them does not move the bars. The scanner now also checks big-endian 16-bit
values, since Neo Geo counters are high-byte-first; treat any discovered map
for this romset as unverified until it is replay-checked. Vision therefore
remains the source of truth, and the `approach` reward term (which needs X
positions) stays inactive for this game.

`hue` matters here. SS2's bars are pure green (~120–137°), while the damage
trail behind them animates through orange and yellow (37–58°) and the stage
lightning is cyan/blue. Keeping only green rejects all three; with the default
wider window the bar edge flickers by up to ~20 px whenever the trail flashes
yellow.

Rounds are detected without a timer: when both bars refill after at least two
damaged frames, the round ended (KO or time-out) and the next one started, so
the episode terminates and the winner is taken from the health right before
the refill. Full-screen hit flashes and HUD-less transition frames are skipped
with the last valid reading kept, so they cannot fake a double KO. A round
lasts about 4,470 emulated frames here (~75 s of game time, comfortably under
the 6,000-frame cap); if neither fighter took damage, the bars are full before
and after the transition, so the refill is invisible and the episode simply
ends at the frame cap.

Throughput is ~9 env steps/s per process (~7 steps/s with video recording),
i.e. a full round takes about 40 s of wall-clock time. Use 4–8 parallel envs
for training.

### Skills and raw inputs together (`action_mode: "hybrid"`)

The real-game configs default to `"hybrid"`: `Discrete(62)` on Samurai Shodown
II, where the first 45 actions are raw stick+button windows and the remaining
17 are scripted skills (movement, normals, jump-ins and the motion inputs
`qcf_A`, `qcf_B`, `dp_A`, `qcb_A`).

Both halves were measured on the ROM from the shipped start state, fighters at
max range, opponent idle. Raw inputs, 350 actions each (~1,050 frames):

| action | damage dealt |
|---|---|
| `right+B` (walk in, tap slash) | 0.281 |
| `left+B` (walk away, tap slash) | 0.000 |
| `neutral+B` (slash in place) | 0.000 |

Skills, 25 actions each — a slash never reaches from that distance, so the
motion hits can only be the projectile:

| action | damage dealt |
|---|---|
| `qcf_A` (↓ ↘ → + A) | 0.148 |
| `qcf_B` (↓ ↘ → + B) | 0.180 |
| `dp_A` (→ ↓ ↘ + A) | 0.148 |
| `qcb_A` (↓ ↙ ← + A) | 0.000 |

`qcb_A` is kept in the library anyway and simply earns a zero: the agent
learns to drop it. Skill names describe the *input*, not a named special-move,
and only `qcf`/`dp` were confirmed to fire here.

Add per-character skills under `special_macros` in the game JSON; extend the
raw half with simultaneous presses (`"primitive_buttons": [["A", "B"]]`) when
a game uses them for throws or evasions.

## Environment API

```python
from fightrl import make_env

env = make_env("sf2_snes", opponent="aggressive")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

* **Observation** — `Dict`
  * `pixels`: `uint8 (4, 84, 84)` grayscale, 4-frame stack (resized from the native frame)
  * `state`: `float32 (12,)` — `[p1_hp, p2_hp, hp_lead, timer, p1_x, p2_x, distance, facing, valid, round_over, 0, 0]`
* **Action space** — depends on `action_mode`
  * `"macro"` (default when the config defines macros): `Discrete(n_macros)` —
    each action is a short scripted input sequence (walk, jump, block, each
    normal, and game-specific specials such as `hadoken_punch`). Macros are
    resolved **relative to facing**, so `fwd` always means "towards the
    opponent". This is how motion inputs (↓ ↘ → + punch) become learnable.
  * `"hybrid"`: `Discrete(n_primitives + n_macros)` — the macro skill library
    **plus** raw stick+button primitives, so the agent can either play a named
    skill or hand-steer for one `frameskip` window. Primitives are
    `9 directions × button sets`; the default button sets are "no button" and
    each single button (add simultaneous presses with `primitive_buttons`).
    Primitive directions are absolute (screen left/right) like a real stick —
    facing-relative motions are the skills' job. Buttons are released on the
    last frame of the window because attacks edge-trigger: a button held every
    frame fires once and is then ignored. Real-game configs ship with this
    mode; use `--action-mode macro` to fall back. Because primitives are
    screen-absolute (as on a real pad), only the facing-relative skills
    transfer when a checkpoint is re-used on the other side of the screen.
  * `"full"`: `MultiDiscrete([9, 2, 2, ...])` — 8-way stick plus one bit per
    button, held for `frameskip` frames. Lower-level but much harder to
    explore with; useful for fine-tuning.
* **Reward** — per game frame, scaled by max health:

  ```
  r = w_deal·damage_dealt − w_take·damage_taken
    + w_lead·Δ(health_lead)          # potential-based shaping (policy-invariant)
    + w_approach·Δ(distance)/W       # closes the gap instead of whiffing at range
    − w_time·frame/fps
    + w_win / −w_loss  on round end
  ```

  Weights live under `"reward"` in the game config. `info` also reports
  `dmg_dealt`, `dmg_taken`, `p1_hp`, `p2_hp`, `winner`, `frames`,
  `state_source`.
* **Episodes** — one round per episode. Ends on KO, timer, or
  `max_episode_frames` (truncation). Each reset reloads the start state, so the
  game never drifts (no menu handling needed).

## Opponents and self-play

`--opponent` accepts:

| Spec | Meaning |
|---|---|
| `aggressive` / `defensive` / `random` / `passive` | scripted bots |
| `checkpoint:path/to/model.zip` | a frozen PPO policy |
| `league:dir` | sample a frozen policy from `dir/*.zip` every episode, with 25% of episodes still played against a scripted style; falls back to the scripted mix until the first checkpoint exists |
| `none` | dummy inputs (useful for pure damage-farming tests) |

Policy opponents see the same pixels but a **mirrored state vector** (their
health in slot 0, facing flipped), so a P1-trained network can play P2.

Recommended curriculum for a new ROM:

1. `--opponent aggressive --steps 200k` — learn to move and hit.
2. `--opponent league:<dir> --steps 2M+` — self-play; the league keeps the
   policy from overfitting to one exploitable bot, while the scripted slice
   keeps chasing, spacing and defense sharp.
3. Evaluate a snapshot against all four scripted styles; keep the model that
   is best *on average*, not the one that farms one bot.

## Performance

Measured on 16 cores + RTX 2080 Ti:

| Setup | Throughput |
|---|---|
| mock, 1 env, no training | ~2,100 env steps/s (~26k game frames/s) |
| mock, 8 processes, PPO on GPU | ~690 env steps/s end-to-end (~8k frames/s) |
| libretro (fbneo, SamSho II, vision, 1 env) | ~9 env steps/s (~27 emulated frames/s) |
| libretro (snes9x, headless) | core-bound; 8 processes is usually enough to saturate a GPU |

`--frameskip` and macro length control the frames-per-step ratio: a step runs
10–20 game frames with macros, so 100k steps ≈ 1–2M frames.

### Reference results (mock game)

`runs/mock4/final.zip` was trained for ~475k additional steps on top of an
earlier league checkpoint (~803k total), resuming with:

```bash
python -m fightrl.train --game mock --league runs/mock4/league \
    --resume runs/mock3/checkpoints/ppo_000325712.zip \
    --steps 475000 --n-envs 8 --device cuda --out runs/mock4
```

20 evaluation episodes per opponent (deterministic):

| Opponent | Win rate | Damage dealt (max 1.0) | Damage taken |
|---|---|---|---|
| aggressive | 1.00 | 1.00 | 0.27 |
| random | 1.00 | 1.00 | 0.35 |
| defensive | 1.00 | 1.00 | 0.30 |
| passive (stationary blocker) | 0.95 | 0.99 | 0.00 |

The passive row matters: an earlier policy that only punished approaching
opponents drew every round against a blocker. The stronger approach shaping
plus the scripted slice of the league (`passive` included) taught it to walk
into range and break the guard.

## Repository layout

```
fightrl/
  env.py               FightEnv (Gymnasium) + observation/action encoding
  train.py             PPO training + league checkpoints (CLI)
  evaluate.py          win-rate evaluation + mp4 replay (CLI)
  ramscan.py           health/timer RAM address discovery (CLI)
  capture.py           keyboard play + save-state capture (CLI)
  opponents.py         scripted bots, frozen policies, self-play league
  backends/
    libretro.py        ctypes libretro frontend (video/input/RAM/save states)
    mock.py            built-in 2D fighter + RAM shadow + HUD renderer
  core/
    controller.py      ControllerState, 8-way directions, facing-relative tokens
    actions.py         macro actions
    state.py           GameState, RAM reader, vision health-bar reader
    config.py          game configs, reward defaults, core resolution
  games/               mock / sf2_snes / sf2_genesis / samsho2_neogeo
tests/                 env, state readers, RAM scanner, libretro, configs
scripts/               fetch_cores.sh, demo_mock.sh
```

## Game config reference

```jsonc
{
  "id": "sf2_snes",
  "backend": "libretro",
  "core": "snes9x_libretro.so",        // name in ./cores or an absolute path
  "rom": "roms/sf2turbo.sfc",
  "system_dir": "cores",               // BIOS / SRAM directory
  "start_state": "states/sf2_snes.state",
  "screen": {"width": 256, "height": 224},
  "frameskip": 3,
  "state_source": "auto",              // auto | ram | vision
  "button_map": {"UP": 4, "LP": 1},    // logical name -> libretro joypad id
  "action_buttons": ["LP", "MP", "HP", "LK", "MK", "HK"],
  "action_mode": "hybrid",              // macro | hybrid | full (default: macro when macros exist)
  "primitive_buttons": [["LP", "HP"]],   // extra button sets for hybrid primitives (optional)
  "special_macros": [ /* motion-input macros appended to the defaults */ ],
  "ram_map": { "p1_hp": {"addr": 0, "width": 1}, "p1_hp_max": 176 },
  "hud": {"p1_bar": [x, y, w, h], "p2_bar": [x, y, w, h], "hue": [35, 150]},  // optional vision override
  "reward": {"deal": 1.0, "take": 1.0, "win": 1.0, "loss": 1.0, "time": 0.0, "lead": 0.25},
  "max_episode_frames": 6000
}
```

libretro joypad ids: `B=0, Y=1, SELECT=2, START=3, UP=4, DOWN=5, LEFT=6,
RIGHT=7, A=8, X=9, L=10, R=11`. The mappings shipped for SF2-SNES,
SF2-Genesis and Neo Geo follow the standard libretro layouts — if a button
feels wrong in `fightrl.capture`, that's the one line to change.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `libretro core ... not found` | run `scripts/fetch_cores.sh`, or set `FIGHRL_CORES=/path/to/cores` |
| `core refused to load ...` | ROM name/format mismatch (e.g. `.sfc` vs `.smc`); FBNeo needs an exact romset name and `neogeo.zip` in `system_dir` |
| `could not read game state` | health bars not found: add `hud` rectangles, or run `fightrl.ramscan` |
| Vision health looks noisy | use a RAM map; vision is ±1–2% because of bar pixel rounding |
| `retro_unserialize failed` | start state was made with a different core version/ROM — recapture |
| Boots to a black screen | the ROM boots into menus; use `fightrl.capture` to record a start state |
| Slow training | more `--n-envs` (4–8 for real games), keep PPO on GPU; vision reads are the main CPU cost — a RAM map removes them |
| `--opponent league:` never switches | checkpoints are written to `runs/<run>/league`; pass that exact path |

## Limitations / next steps

* Only health, timer and (optionally) x-positions are read from the game; there
  is no round-win tracking across a full match, no combo counter, no
  block/whiff detection.
* Episodes are single rounds and the agent is always P1.
* Special moves are wrapped only where a motion macro is configured (SF2 ships
  with hadouken/tatsumaki/shoryuken, Samurai Shodown II with quarter-circle,
  dragon-punch and reverse quarter-circle slashes); add your own under
  `special_macros`. In `"hybrid"` mode those skills sit alongside raw stick
  inputs, so the agent can still micro-manage movement. The shipped motions
  are verified to fire at range on Samurai Shodown II's start state (see the
  table above), but each character's full move list is not mapped — a mirror
  match of Haohmaru is all that has been probed.
* Character selection is bypassed by the start state; to train multiple
  matchups, record one start state per matchup and train separate runs.
* `state_source: "vision"` assumes the health bars are vivid and in the top
  30% of the screen (true for SF2 and Samurai Shodown II, and for the mock).
* Samurai Shodown II currently has no reliable RAM map, so it runs on vision
  only; the HUD rectangles and hue window are tuned in its game config.
