# -*- coding: utf-8 -*-
"""
Download additional climate-zone gauge networks for transfer experiments:
- willamette: humid Pacific Northwest
- animas: snowmelt Rockies
- verde: semi-arid Southwest

Reuses Potomac/James if already present.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"
END = "2023-12-31"

NETWORKS = {
    "willamette": {
        "climate": "humid_pacific_nw",
        "target": "14191000",
        "upstream": ["14174000", "14182500"],
        "sites": {
            "14174000": {"name": "Willamette River at Harrisburg, OR", "lat": 44.2740, "lon": -123.1726, "role": "midstream"},
            "14182500": {"name": "Santiam River at Jefferson, OR", "lat": 44.7173, "lon": -123.0112, "role": "tributary"},
            "14191000": {"name": "Willamette River at Salem, OR", "lat": 44.9443, "lon": -123.0426, "role": "downstream"},
        },
    },
    "animas": {
        "climate": "snowmelt_rockies",
        "target": "09363500",
        "upstream": ["09358000", "09361500"],
        "sites": {
            "09358000": {"name": "Cement Creek at Silverton, CO", "lat": 37.8197, "lon": -107.6631, "role": "tributary"},
            "09361500": {"name": "Animas River below Silverton, CO", "lat": 37.7950, "lon": -107.6673, "role": "midstream"},
            "09363500": {"name": "Animas River at Durango, CO", "lat": 37.2794, "lon": -107.8803, "role": "downstream"},
        },
    },
    "verde": {
        "climate": "semiarid_southwest",
        "target": "09506000",
        "upstream": ["09504000", "09503700"],
        "sites": {
            "09504000": {"name": "Verde River near Clarkdale, AZ", "lat": 34.7567, "lon": -112.0549, "role": "midstream"},
            "09503700": {"name": "Oak Creek near Cornville, AZ", "lat": 34.7170, "lon": -111.8940, "role": "tributary"},
            "09506000": {"name": "Verde River near Camp Verde, AZ", "lat": 34.5903, "lon": -111.9085, "role": "downstream"},
        },
    },
}


def _get_json(url: str, retries: int = 8, sleep0: float = 10.0) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 429:
                wait = sleep0 * (2 ** i)
                print(f"  rate-limited, sleep {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            wait = sleep0 * (1.5 ** i)
            print(f"  retry: {e}; sleep {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(last)


def fetch_usgs(site_id: str) -> pd.DataFrame:
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json&sites={site_id}&parameterCd=00060"
        f"&startDT={START}&endDT={END}&statCd=00003"
    )
    values = _get_json(url)["value"]["timeSeries"][0]["values"][0]["value"]
    rows = []
    for v in values:
        val = float(v["value"]) if v["value"] != "-999999" else np.nan
        rows.append({"date": pd.to_datetime(v["dateTime"]).normalize(), "discharge_cfs": val})
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date")


def fetch_meteo(lat: float, lon: float) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={START}&end_date={END}"
        "&daily=precipitation_sum,temperature_2m_mean&timezone=UTC"
    )
    d = _get_json(url)["daily"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "precip_mm": d["precipitation_sum"],
            "temp_c": d["temperature_2m_mean"],
        }
    )


def build_wide(key: str) -> pd.DataFrame:
    net = NETWORKS[key]
    out = DATA_DIR / f"{key}_daily.csv"
    if out.exists() and len(pd.read_csv(out)) > 5000:
        print(f"[{key}] skip, existing {len(pd.read_csv(out))} rows")
        return pd.read_csv(out, parse_dates=["date"])

    frames = []
    for sid, meta in net["sites"].items():
        print(f"[{key}] {sid} {meta['name']}")
        dq = fetch_usgs(sid)
        time.sleep(2.0)
        mq = fetch_meteo(meta["lat"], meta["lon"])
        m = dq.merge(mq, on="date", how="inner")
        m["site_id"] = sid
        frames.append(m)
        print(f"  rows={len(m)}")
        time.sleep(6.0)

    long = pd.concat(frames, ignore_index=True)
    long.to_csv(DATA_DIR / f"{key}_daily_long.csv", index=False)
    target = net["target"]
    wide = long[long["site_id"] == target][["date", "discharge_cfs", "precip_mm", "temp_c"]].rename(
        columns={"discharge_cfs": "Q_target", "precip_mm": "precip_target", "temp_c": "temp_target"}
    )
    for sid in net["upstream"]:
        sub = long[long["site_id"] == sid][["date", "discharge_cfs", "precip_mm"]].rename(
            columns={"discharge_cfs": f"Q_{sid}", "precip_mm": f"precip_{sid}"}
        )
        wide = wide.merge(sub, on="date", how="left")
    wide = wide.dropna().reset_index(drop=True)
    wide.to_csv(out, index=False)
    print(f"Saved {len(wide)} -> {out.name}")
    return wide


def main():
    meta_path = DATA_DIR / "climate_networks.json"
    existing = {}
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
    existing.update(NETWORKS)
    # also keep potomac/james climate tags if present
    existing.setdefault("potomac", {"climate": "humid_mid_atlantic"})
    existing.setdefault("james", {"climate": "humid_mid_atlantic"})
    meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    for key in NETWORKS:
        try:
            build_wide(key)
        except Exception as e:
            print(f"FAILED {key}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
