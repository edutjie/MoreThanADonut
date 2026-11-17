"""Read official tactical automatic tracking in metres."""


import numpy as np


from games import GAME, GAMES, require, source_frames


from starter.baseline import team_of_each_track


from tactical_baseline import read_tracking


def contest_tracking(data_dir):
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=data_dir)
    teams = team_of_each_track(meta)
    pitch = np.array(GAMES[GAME]["pitch"], dtype=np.float32)
    return read_tracking(raw, source_frames(), len(teams), pitch), teams, pitch, GAMES[GAME]["fps"]
