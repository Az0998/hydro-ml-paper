# -*- coding: utf-8 -*-
"""Download USGS streamflow and Open-Meteo precipitation for study basins."""

import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"

# Potomac River basin: upstream -> downstream network (spatial topology for paper)
BASINS = {
    "01636500": {
        "name": "Shenandoah River at Millville, WV",
        "lat": 39.2886,
        "lon": -77.7867,
        "role": "tributary",
    },
    "01638480": {
        "name": "Potomac River at Harpers Ferry, WV",
        "lat": 39.3226,
        "lon": -77.7283,
        "role": "midstream",
    },
    "01646500": {
        "name": "Potomac River at Washington, DC",
        "lat": 38.9498,
        "lon": -77.1278,
        "role": "downstream",
    },
}

START_DATE = "2000-01-01"
END_DATE = "2023-12-31"


def fetch_usgs_discharge(site_id: str) -> pd.DataFrame:
    """Fetch daily mean discharge (cfs) from USGS NWIS."""
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json&sites={site_id}&parameterCd=00060"
        f"&startDT={START_DATE}&endDT={END_DATE}"
        "&statCd=00003"
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    js = r.json()

    ts = js["value"]["timeSeries"][0]
    values = ts["values"][0]["value"]
    rows = []
    for v in values:
        val = float(v["value"]) if v["value"] != "-999999" else float("nan")
        rows.append({"date": pd.to_datetime(v["dateTime"]).normalize(), "discharge_cfs": val})

    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    df["site_id"] = site_id
    return df


def fetch_open_meteo_precip(lat: float, lon: float, site_id: str) -> pd.DataFrame:
    """Fetch daily precipitation (mm) from Open-Meteo archive API."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&daily=precipitation_sum,temperature_2m_mean"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    daily = r.json()["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "precip_mm": daily["precipitation_sum"],
            "temp_c": daily["temperature_2m_mean"],
        }
    )
    df["site_id"] = site_id
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_discharge = []
    all_meteo = []

    for site_id, meta in BASINS.items():
        print(f"Downloading {site_id}: {meta['name']}")
        try:
            dq = fetch_usgs_discharge(site_id)
            mq = fetch_open_meteo_precip(meta["lat"], meta["lon"], site_id)
            all_discharge.append(dq)
            all_meteo.append(mq)
            print(f"  discharge: {len(dq)} days, meteo: {len(mq)} days")
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(0.5)

    discharge_df = pd.concat(all_discharge, ignore_index=True)
    meteo_df = pd.concat(all_meteo, ignore_index=True)

    # Merge per site
    merged = discharge_df.merge(meteo_df, on=["date", "site_id"], how="inner")
    merged = merged.sort_values(["site_id", "date"]).reset_index(drop=True)

    # Pivot to wide format for multi-station modeling
    target_site = "01646500"
    wide = merged[merged["site_id"] == target_site][["date", "discharge_cfs", "precip_mm", "temp_c"]].rename(
        columns={
            "discharge_cfs": "Q_target",
            "precip_mm": "precip_target",
            "temp_c": "temp_target",
        }
    )

    for site_id in ["01636500", "01638480"]:
        sub = merged[merged["site_id"] == site_id][["date", "discharge_cfs", "precip_mm"]]
        sub = sub.rename(columns={"discharge_cfs": f"Q_{site_id}", "precip_mm": f"precip_{site_id}"})
        wide = wide.merge(sub, on="date", how="left")

    wide = wide.dropna().reset_index(drop=True)

    merged.to_csv(DATA_DIR / "basin_daily_long.csv", index=False)
    wide.to_csv(DATA_DIR / "potomac_daily.csv", index=False)

    meta_out = {k: v for k, v in BASINS.items()}
    (DATA_DIR / "basin_metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    print(f"\nSaved {len(wide)} daily records -> {DATA_DIR / 'potomac_daily.csv'}")
    print(f"Date range: {wide['date'].min()} to {wide['date'].max()}")


if __name__ == "__main__":
    main()
