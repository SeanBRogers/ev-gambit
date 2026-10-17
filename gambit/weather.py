"""
Kickoff weather from Open-Meteo (free, no key). Only wind enters the model, because
wind is the environmental with an estimated, significant effect in the fitted data
(pass share and yards/attempt). Temperature and precipitation are shown for context.

Coordinates are stadium-level approximations; good enough for a weather grid cell.
"""

from __future__ import annotations

import requests

HOME_STADIUM = {
    "ARI": (33.528, -112.263), "ATL": (33.755, -84.401), "BAL": (39.278, -76.623),
    "BUF": (42.774, -78.787), "CAR": (35.226, -80.853), "CHI": (41.862, -87.617),
    "CIN": (39.095, -84.516), "CLE": (41.506, -81.700), "DAL": (32.747, -97.094),
    "DEN": (39.744, -105.020), "DET": (42.340, -83.046), "GB": (44.501, -88.062),
    "HOU": (29.685, -95.411), "IND": (39.760, -86.164), "JAX": (30.324, -81.637),
    "KC": (39.049, -94.484), "LV": (36.091, -115.183), "LAC": (33.953, -118.339),
    "LA": (33.953, -118.339), "MIA": (25.958, -80.239), "MIN": (44.974, -93.258),
    "NE": (42.091, -71.264), "NO": (29.951, -90.081), "NYG": (40.814, -74.074),
    "NYJ": (40.814, -74.074), "PHI": (39.901, -75.168), "PIT": (40.447, -80.016),
    "SF": (37.403, -121.970), "SEA": (47.595, -122.332), "TB": (27.976, -82.503),
    "TEN": (36.166, -86.771), "WAS": (38.908, -76.865),
}
NEUTRAL_SITE = {
    "LON00": (51.556, -0.280), "LON01": (51.604, -0.066), "LON02": (51.456, -0.342),
    "PAR00": (48.924, 2.360), "MUN00": (48.219, 11.625), "MUN01": (48.219, 11.625),
    "FRA00": (50.069, 8.646), "MAD00": (40.453, -3.688), "DUB00": (53.361, -6.251),
    "MEL00": (-37.820, 144.983), "SAO00": (-23.545, -46.474), "BER00": (52.515, 13.239),
    "RIO00": (-22.912, -43.230), "MAD01": (40.453, -3.688), "MEX00": (19.303, -99.150),
}
INDOOR = {"dome", "closed"}
# Retractable/covered venues: nflverse leaves `roof` blank for future games here. Default to
# closed (wind 0) unless the user knows otherwise.
RETRACTABLE = {"ATL97", "IND00", "HOU00", "PHO00", "DAL00", "MAD01"}


def stadium_coords(home_team: str, stadium_id: str | None, home_stadium_id: str | None = None):
    """Neutral sites by id; the home team's coordinates only when the game is at its usual
    stadium. An unrecognized venue returns None rather than silently using the home city."""
    if stadium_id and stadium_id in NEUTRAL_SITE:
        return NEUTRAL_SITE[stadium_id]
    if home_stadium_id is not None and stadium_id and stadium_id != home_stadium_id:
        return None
    return HOME_STADIUM.get(home_team)


def kickoff_weather(home_team: str, stadium_id: str | None, gameday: str, gametime: str,
                    home_stadium_id: str | None = None, timeout: float = 8.0) -> dict:
    """Forecast at the kickoff hour (gametime is US Eastern in nflverse)."""
    coords = stadium_coords(home_team, stadium_id, home_stadium_id)
    if coords is None:
        return {"error": f"unrecognized venue {stadium_id} — enter wind manually"}
    lat, lon = coords
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=timeout, params=dict(
            latitude=lat, longitude=lon, hourly="wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation",
            wind_speed_unit="mph", temperature_unit="fahrenheit", precipitation_unit="inch",
            timezone="America/New_York", start_date=gameday, end_date=gameday))
        r.raise_for_status()
        h = r.json()["hourly"]
    except Exception as e:  # network blocked, date out of forecast range, etc.
        return {"error": str(e)[:160]}
    hour = f"{gameday}T{gametime[:2]}:00"
    if hour not in h["time"]:
        return {"error": f"kickoff hour {hour} not in forecast"}
    i = h["time"].index(hour)
    return dict(wind_mph=h["wind_speed_10m"][i], gust_mph=h["wind_gusts_10m"][i],
                temp_f=h["temperature_2m"][i], precip_in=h["precipitation"][i])
