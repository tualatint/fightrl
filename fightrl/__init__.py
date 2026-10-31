"""FightRL: reinforcement-learning environments for 2D fighting games.

Two backends are provided:

* ``libretro`` -- a generic libretro frontend written with ctypes.  Works with
  any libretro core (snes9x, genesis_plus_gx, fbneo, ...) so it runs real
  Street Fighter II / Samurai Shodown II ROM dumps that you own.
* ``mock`` -- a deterministic, dependency-free 2D fighter implemented in
  Python.  It renders a Street-Fighter-style HUD and exposes a game-RAM
  image, so the whole pipeline (state readers, reward, RAM scanner, training,
  self-play) can be developed and tested without any copyrighted content.

No ROMs are bundled or downloaded by this package.
"""

from .core.config import GameConfig, load_game_config
from .env import FightEnv, make_env

__all__ = ["FightEnv", "make_env", "GameConfig", "load_game_config"]
__version__ = "0.1.0"
