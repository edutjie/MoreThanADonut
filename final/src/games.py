'Per-game constants shared by the evaluation scripts and the starter tooling.\n\nThis module deliberately imports nothing beyond the standard library, so it can be loaded by any\ninterpreter and by code that has no numpy or pandas available.\n\nThe alignment is per half: id_frame = video_frame + OFFSETS[half]. The two halves genuinely\ndisagree, so no single integer offset is frame-accurate for both, and frame accuracy is what the\nevaluation in evaluation/ needs -- a one-frame error moves every transition it scores.\n\nEverything a submission contains is indexed by *video* frame. The alignment below only matters\nwhen you read the tracking data, which runs on its own counter.\n\nData location\n-------------\nThe match assets are gated: they are not in this repository. Download them (see DATA.md) and put\nthem in ./data, or point MTAH_DATA_DIR somewhere else:\n\n    export MTAH_DATA_DIR=/path/to/mtah2027        # macOS / Linux\n    $env:MTAH_DATA_DIR = "D:\\mtah2027"           # Windows PowerShell\n\nThe evaluation itself needs none of this -- it reads only CSVs -- so a missing data directory is\nnot an error here.'


import os


GAME = "BAR-ATH"


DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")


GAMES = {
    "BAR-ATH": {
        # The match comes from two cameras, each with its own folder, its own stem and its own
        # timeline -- see SOURCES below and asset_paths(). The tactical video is the frame index
        # authority for this challenge: row i of a submission is frame i of that file. The
        # broadcast recording is an optional extra input and is NOT aligned to it.
        #
        # n_video_frames, fps and pitch describe the TACTICAL source, because that is the
        # submission contract; per-source counts live in SOURCES.
        "n_video_frames": 199319,
        "fps": 25,
        # id_frame = video_frame + offsets[half], measured frame by frame against the video.
        #
        # The halves differ by one frame. A single offset therefore cannot be exact for both,
        # which is why this is a per-half table and not a scalar. It started life as one number
        # (1702056) derived from the first-half kickoff, which was good enough for watching the
        # video and not good enough for scoring transition timing to the frame.
        "offsets": {1: 1702057, 2: 1702058},
        # IdFrame span of each half, read off the two contiguous blocks in the tracking data (the
        # only gap is halftime). These are what decide which offset applies to a frame, so they
        # are data, not decoration.
        "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
        # Kickoff anchors, kept as convenient playback start positions. They are observations
        # about the video -- "the ball is struck on this frame" -- so they do not move when the
        # offsets are corrected, and they do not feed the offsets.
        "kickoff": {"minute": 7, "second": 27, "frame": 11},
        "kickoff_2nd": {"minute": 75, "second": 49, "frame": 16},
        # Real pitch size in metres.
        "pitch": (105.3, 68.0),
    },
}


SOURCES = {
    "tactical": {
        "stem": "BAR-ATH_tactical",
        "n_video_frames": 199319,
        # Smart Tagging is timed against the tactical video, so it belongs to this source only.
        "assets": ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"),
    },
    "broadcast": {
        "stem": "BAR-ATH_broadcast",
        "n_video_frames": 164488,
        "assets": ("video", "atd_meta", "atd_raw", "homography"),
    },
}


DEFAULT_SOURCE = "tactical"


SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}


def asset_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Every provided file for one source of a game, keyed by asset, under DATA_DIR.

    The bundle is one folder per recording:

        data/
        |-- tactical/                           the evaluation reference
        |   |-- BAR-ATH_tactical.mp4
        |   |-- BAR-ATH_tactical_atd.xml        FIFA/EPTS metadata
        |   |-- BAR-ATH_tactical_atd.txt        FIFA/EPTS raw tracking, players and ball
        |   |-- BAR-ATH_tactical_homography.csv
        |   `-- BAR-ATH_tactical_smart_tagging.xml
        `-- broadcast/                          optional, NOT aligned to tactical
            |-- BAR-ATH_broadcast.mp4
            |-- BAR-ATH_broadcast_atd.xml
            |-- BAR-ATH_broadcast_atd.txt
            `-- BAR-ATH_broadcast_homography.csv

    Defaults to the tactical source, because that is the one the submission is indexed to and the
    one every existing caller means. Broadcast is opt-in:

        asset_paths()                     # tactical
        asset_paths(source="broadcast")   # broadcast -- no smart_tagging key, it has none

    The paths are not checked for existence -- see require() for that.
    """
    root = data_dir or DATA_DIR
    entry = _source(game, source)
    folder = os.path.join(root, source)
    return {name: os.path.join(folder, entry["stem"] + SUFFIXES[name])
            for name in entry["assets"]}


def _source(game, source):
    """The SOURCES entry, or a readable error naming the ones that exist."""
    if source not in SOURCES:
        raise KeyError("no such source {!r}; have {}".format(source, sorted(SOURCES)))
    if game not in GAMES:
        raise KeyError("no such game {!r}; have {}".format(game, sorted(GAMES)))
    return SOURCES[source]


def source_frames(source=DEFAULT_SOURCE, game=GAME):
    """How many video frames that source has. Tactical is the submission length."""
    return _source(game, source)["n_video_frames"]


def require(names, game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Paths for the named assets, or one readable error naming every missing file.

    The match data is gated and is not in the repository, so "it is not there" is the normal
    first-run state rather than a bug. Saying which files and where they were expected is the
    difference between a two-second fix and a support message.
    """
    paths = asset_paths(game, data_dir, source)
    unknown = [name for name in names if name not in paths]
    if unknown:
        raise KeyError("source {!r} has no asset {}; it provides {}".format(
            source, unknown, sorted(paths)))

    wanted = [(name, paths[name]) for name in names]
    missing = [path for _, path in wanted if not os.path.isfile(path)]
    if missing:
        raise IOError(
            "match data not found:\n  {}\n\nThe data is gated -- see DATA.md. Unpack the bundle "
            "into\n  {}\nor point MTAH_DATA_DIR at wherever you put it."
            .format("\n  ".join(missing), data_dir or DATA_DIR))
    return [path for _, path in wanted]


def halves(game):
    """[(half, first_id_frame, last_id_frame)] in play order."""
    table = GAMES[game]["halves"]
    return [(half, table[half][0], table[half][1]) for half in sorted(table)]


def frame_offset(game, half, adjust=0):
    """Offset such that id_frame = video_frame + offset, for one half."""
    return GAMES[game]["offsets"][half] + adjust


def half_video_blocks(game, adjust=0):
    """[(half, first_video_frame, last_video_frame)], each half mapped by its own offset.

    For BAR-ATH: half 1 covers video 11185..90651 and half 2 covers 110808..190197, leaving
    20156 video frames of halftime between them. The blocks are far apart, so a video frame
    belongs to at most one of them and video_to_id needs no tie-breaking.
    """
    blocks = []
    for half, first_id, last_id in halves(game):
        offset = frame_offset(game, half, adjust)
        blocks.append((half, first_id - offset, last_id - offset))
    return blocks
