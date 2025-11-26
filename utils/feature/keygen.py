from itertools import product

__all__ = ["gen_keys"]

def gen_keys(row: dict, window_tag: str):
    """Yield cache keys for batter & pitcher under various granularities."""
    batter   = row["batter"]
    pitcher  = row["pitcher"]
    throws   = row.get("pitcher_hand")
    pitch_tp = row.get("pitch_type")

    # base game-level key
    yield ("batter", batter, "game", window_tag, "all")

    for gran, versus in product(
        ("pitch", "pa", "game"),
        ("all", "vs_hand", "vs_pitcher", "vs_bullpen")
    ):
        yield ("batter", batter, gran, window_tag, versus)
        yield ("pitcher", pitcher, gran, window_tag, versus) 