import fastf1
import pandas as pd
import requests

# Enable local cache for faster repeated runs
fastf1.Cache.enable_cache("cache")

JOLPICA_BASE = "http://api.jolpi.ca/ergast/f1"

def get_event_schedule(year: int) -> pd.DataFrame:
    """Get the season schedule as a DataFrame."""
    sch = fastf1.get_event_schedule(year)
    return sch.reset_index(drop=True)

def _driver_standings_before(year: int, round_number: int) -> pd.DataFrame:
    """Get driver standings BEFORE a given round using Jolpica."""
    if round_number <= 1:
        return pd.DataFrame(columns=["code", "driver_points_before"])
    url = f"{JOLPICA_BASE}/{year}/{round_number-1}/driverStandings.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    tables = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not tables:
        return pd.DataFrame(columns=["code", "driver_points_before"])
    drivers = []
    for entry in tables[0]["DriverStandings"]:
        code = entry["Driver"].get("code")
        if code:
            drivers.append({"code": code, "driver_points_before": float(entry["points"])})
    return pd.DataFrame(drivers)

def load_race_dataframe(year: int, event_name: str) -> pd.DataFrame:
    """Load race data for one event."""
    race = fastf1.get_session(year, event_name, "R")
    race.load(laps=True, telemetry=False, weather=False, messages=False)

    res = race.results.copy()
    res = res.rename(columns={
        "Abbreviation": "code",
        "TeamName": "team",
        "GridPosition": "grid_position",
        "Position": "final_position"
    })

    laps = race.laps
    lap_feats = (
        laps.groupby("DriverNumber", as_index=False)
            .agg(
                avg_lap_ms=("LapTime", lambda s: s.dropna().dt.total_seconds().mean() * 1000),
                fastest_lap_ms=("LapTime", lambda s: s.dropna().dt.total_seconds().min() * 1000),
                total_laps=("LapNumber", "max")
            )
    )

    df = res.merge(lap_feats, on="DriverNumber", how="left")
    df["year"] = year
    df["event"] = event_name
    df["top10"] = (df["final_position"] <= 10).astype(int)

    rnd = int(race.event["RoundNumber"])
    standings = _driver_standings_before(year, rnd)
    if not standings.empty:
        df = df.merge(standings, on="code", how="left")
    else:
        df["driver_points_before"] = None

    return df[[
        "year", "event", "DriverNumber", "code", "team",
        "grid_position", "driver_points_before",
        "avg_lap_ms", "fastest_lap_ms", "total_laps",
        "final_position", "top10"
    ]]

def build_dataset(year_start: int = 2018, year_end: int = 2025) -> pd.DataFrame:
    """Loop through seasons/events and build a dataset."""
    frames = []
    for yr in range(year_start, year_end + 1):
        sch = get_event_schedule(yr)
        for _, row in sch.iterrows():
            try:
                frames.append(load_race_dataframe(yr, row["EventName"]))
            except Exception as e:
                print(f"[WARN] {yr} {row['EventName']}: {e}")
                continue
    return pd.concat(frames, ignore_index=True)
