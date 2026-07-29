# -*- coding: utf-8 -*-
"""
Download:
1) Extended Potomac daily discharge/meteo through 2024
2) Real QPF precip leads (Open-Meteo Previous Runs, hourly -> daily) for Potomac target
3) Neighboring James River gauge network (same mid-Atlantic setting)
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
END = "2024-12-31"
QPF_START = "2024-01-01"  # Previous-runs precip reliably available from 2024

NETWORKS = {
    "potomac": {
        "target": "01646500",
        "upstream": ["01636500", "01638480"],
        "sites": {
            "01636500": {"name": "Shenandoah River at Millville, WV", "lat": 39.2886, "lon": -77.7867, "role": "tributary"},
            "01638480": {"name": "Potomac River at Harpers Ferry, WV", "lat": 39.3226, "lon": -77.7283, "role": "midstream"},
            "01646500": {"name": "Potomac River at Washington, DC", "lat": 38.9498, "lon": -77.1278, "role": "downstream"},
        },
    },
    "james": {
        "target": "02037500",
        "upstream": ["02035000", "02034000"],
        "sites": {
            "02035000": {"name": "James River at Cartersville, VA", "lat": 37.6707, "lon": -78.0858, "role": "midstream"},
            "02034000": {"name": "Rivanna River at Palmyra, VA", "lat": 37.8632, "lon": -78.2650, "role": "tributary"},
            "02037500": {"name": "James River near Richmond, VA", "lat": 37.5632, "lon": -77.5472, "role": "downstream"},
        },
    },
}


def _get_json(url: str, retries: int = 6, sleep0: float = 8.0) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 429:
                wait = sleep0 * (2 ** i)
                print(f"  rate-limited, sleep {wait:.0f}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            wait = sleep0 * (1.5 ** i)
            print(f"  retry after error: {e}; sleep {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(last)


def fetch_usgs(site_id: str) -> pd.DataFrame:
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json&sites={site_id}&parameterCd=00060"
        f"&startDT={START}&endDT={END}&statCd=00003"
    )
    js = _get_json(url)
    values = js["value"]["timeSeries"][0]["values"][0]["value"]
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


def fetch_qpf_daily(lat: float, lon: float, leads=(1, 3, 7)) -> pd.DataFrame:
    """
    Real model QPF from Open-Meteo Previous Runs API (hourly), aggregated to daily sums.
    precipitation_previous_dayN = value forecast N days before valid time.
    """
    vars_ = ",".join([f"precipitation_previous_day{n}" for n in leads])
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": QPF_START,
        "end_date": END,
        "hourly": vars_,
        "timezone": "UTC",
        "models": "gfs_seamless",
    }
    # build query string for _get_json
    from urllib.parse import urlencode
    full = f"{url}?{urlencode(params)}"
    h = _get_json(full, retries=8, sleep0=10)["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    for n in leads:
        col = f"precipitation_previous_day{n}"
        df[f"qpf_lead{n}"] = h[col]
    df["date"] = df["time"].dt.normalize()
    daily = df.groupby("date", as_index=False)[[f"qpf_lead{n}" for n in leads]].sum(min_count=1)
    return daily


def build_wide(network_key: str) -> pd.DataFrame:
    net = NETWORKS[network_key]
    target = net["target"]
    out_csv = DATA_DIR / f"{network_key}_daily.csv"
    if out_csv.exists() and network_key != "force":
        # resume: if complete file exists, skip
        existing = pd.read_csv(out_csv, parse_dates=["date"])
        if len(existing) > 8000:
            print(f"[{network_key}] skip download, existing {len(existing)} rows")
            return existing

    frames = []
    for sid, meta in net["sites"].items():
        print(f"[{network_key}] {sid} {meta['name']}")
        dq = fetch_usgs(sid)
        time.sleep(2.0)
        mq = fetch_meteo(meta["lat"], meta["lon"])
        m = dq.merge(mq, on="date", how="inner")
        m["site_id"] = sid
        frames.append(m)
        print(f"  rows={len(m)}")
        time.sleep(5.0)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(DATA_DIR / f"{network_key}_daily_long.csv", index=False)

    wide = long[long["site_id"] == target][["date", "discharge_cfs", "precip_mm", "temp_c"]].rename(
        columns={"discharge_cfs": "Q_target", "precip_mm": "precip_target", "temp_c": "temp_target"}
    )
    for sid in net["upstream"]:
        sub = long[long["site_id"] == sid][["date", "discharge_cfs", "precip_mm"]].rename(
            columns={"discharge_cfs": f"Q_{sid}", "precip_mm": f"precip_{sid}"}
        )
        wide = wide.merge(sub, on="date", how="left")
    wide = wide.dropna().reset_index(drop=True)
    wide.to_csv(out_csv, index=False)
    print(f"Saved {len(wide)} -> {network_key}_daily.csv ({wide['date'].min()} .. {wide['date'].max()})")
    return wide


def main():
    meta = {k: v for k, v in NETWORKS.items()}
    (DATA_DIR / "networks_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pot = build_wide("potomac")
    pot.to_csv(DATA_DIR / "potomac_daily.csv", index=False)

    build_wide("james")

    qpf_path = DATA_DIR / "potomac_qpf_daily.csv"
    if qpf_path.exists() and len(pd.read_csv(qpf_path)) > 300:
        print("QPF exists, skip download")
        qpf = pd.read_csv(qpf_path, parse_dates=["date"])
    else:
        tmeta = NETWORKS["potomac"]["sites"][NETWORKS["potomac"]["target"]]
        print("Downloading GFS previous-run QPF for Potomac target ...")
        time.sleep(5.0)
        qpf = fetch_qpf_daily(tmeta["lat"], tmeta["lon"], leads=(1, 3, 7))
        qpf.to_csv(qpf_path, index=False)
        print(f"Saved QPF {len(qpf)} days -> potomac_qpf_daily.csv")

    pot2 = pot.merge(qpf, on="date", how="left")
    pot2.to_csv(DATA_DIR / "potomac_daily_with_qpf.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
