"""Fetch REAL historical carbon-intensity traces (30 days, hourly) and write the three profile CSVs.

  python scripts/fetch_carbon_data.py --start 2026-09-01          # UK Carbon Intensity API (free, no login)
  python scripts/fetch_carbon_data.py --start 2026-09-01 --import high=my_poland.csv:ENTSOE_PL

Source: UK Carbon Intensity API (regional endpoint). It gives ~half-hourly *forecast* values per region; we average
to hourly. Regions are ranked by mean intensity over the window and the lowest / median / highest become the
low / medium / high profiles (selection is data-driven and recorded in data/carbon_profiles_meta.json).
NOTE: UK regions are not "coal-heavy". If you need a genuinely fossil-heavy grid, import one with --import
(CSV columns: datetime, carbon_intensity_gco2_kwh) and give it a source label.

STATUS: written against the public API docs; it could not be run in the build environment (no network access to
api.carbonintensity.org.uk), so run it once on your machine and check data/carbon_profiles_meta.json."""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.carbonintensity.org.uk"
REGION_IDS = list(range(1, 15))          # England/Scotland/Wales DNO regions (15-17 are aggregates)
H = 30 * 24


def get(url: str):
    for attempt in range(3):
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")


def fetch_region(region_id: int, start: dt.datetime, end: dt.datetime):
    rows, name = [], str(region_id)
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=13), end)
        url = f"{API}/regional/intensity/{cur:%Y-%m-%dT%H:%MZ}/{nxt:%Y-%m-%dT%H:%MZ}/regionid/{region_id}"
        j = get(url)["data"]
        name = j.get("shortname") or j.get("dnoregion") or name
        for item in j["data"]:
            inten = item["intensity"]
            v = inten.get("actual") if inten.get("actual") is not None else inten.get("forecast")
            rows.append({"datetime": item["from"], "v": v})
        cur = nxt
        time.sleep(0.3)
    df = pd.DataFrame(rows).dropna()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    hourly = df.set_index("datetime")["v"].astype(float).resample("1h").mean().dropna()
    return name, hourly.iloc[:H]


def write_profile(profile: str, hourly: pd.Series, source: str):
    out = pd.DataFrame({"datetime": hourly.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "carbon_intensity_gco2_kwh": hourly.round(2).to_numpy(), "source": source})
    out.to_csv(ROOT / "data" / f"carbon_intensity_{profile}.csv", index=False)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC); 30 days are fetched")
    ap.add_argument("--only", default="low,medium,high",
                    help="comma list of profiles to (re)write; the others are left untouched, e.g. --only low")
    ap.add_argument("--import", dest="imports", action="append", default=[],
                    help="profile=path.csv:SOURCE_LABEL  (overrides that profile with your own real trace)")
    a = ap.parse_args()
    start = dt.datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=30)
    meta = {"window_start": start.isoformat(), "window_end": end.isoformat(), "api": API, "regions": {}, "chosen": {}}
    data = {}
    for rid in REGION_IDS:
        try:
            name, hourly = fetch_region(rid, start, end)
            data[rid] = (name, hourly)
            meta["regions"][rid] = {"name": name, "mean_gco2_kwh": float(hourly.mean()), "hours": int(len(hourly))}
            print(f"region {rid:>2} {name:<28} mean={hourly.mean():7.1f}  hours={len(hourly)}")
        except Exception as e:
            print(f"region {rid} failed: {e}", file=sys.stderr)
    if len(data) < 3:
        sys.exit("Fewer than 3 regions fetched; cannot derive low/medium/high profiles.")
    # Skip regions whose reported intensity is ~0 for many hours (e.g. North Scotland): they break the tercile rules.
    usable = {}
    for rid, (name, hourly) in data.items():
        zero = float((hourly <= 0.5).mean())
        if hourly.mean() >= 10 and zero <= 0.05:
            usable[rid] = (name, hourly)
        else:
            meta["regions"][rid]["excluded"] = f"mean={hourly.mean():.1f}, ~zero hours={zero:.0%}"
            print(f"  skipping region {rid} {name}: mean={hourly.mean():.1f}, ~zero hours={zero:.0%}")
    if len(usable) < 3:
        sys.exit("Fewer than 3 usable regions; import traces with --import.")
    ranked = sorted(usable, key=lambda r: usable[r][1].mean())
    pick = {"low": ranked[0], "medium": ranked[len(ranked) // 2], "high": ranked[-1]}
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    old_meta_path = ROOT / "data" / "carbon_profiles_meta.json"
    old_chosen = json.loads(old_meta_path.read_text(encoding="utf-8")).get("chosen", {}) if old_meta_path.exists() else {}
    for prof, rid in pick.items():
        if prof not in only:
            if prof in old_chosen:
                meta["chosen"][prof] = old_chosen[prof]            # keep the earlier choice on record
            print(f"  (left '{prof}' untouched; would have been {usable[rid][0]})")
            continue
        name, hourly = usable[rid]
        n = write_profile(prof, hourly, f"uk_carbon_intensity_api_regional:{name}")
        meta["chosen"][prof] = {"region_id": rid, "region": name, "mean": float(hourly.mean()), "hours": n}
    for spec in a.imports:
        prof, rest = spec.split("=", 1)
        pth, label = rest.rsplit(":", 1)
        df = pd.read_csv(pth)
        s = pd.Series(df["carbon_intensity_gco2_kwh"].astype(float).to_numpy()[:H],
                      index=pd.to_datetime(df["datetime"], utc=True)[:H])
        write_profile(prof, s, label)
        meta["chosen"][prof] = {"imported_from": pth, "source": label, "mean": float(s.mean()), "hours": int(len(s))}
    (ROOT / "data" / "carbon_profiles_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["chosen"], indent=2))


if __name__ == "__main__":
    main()