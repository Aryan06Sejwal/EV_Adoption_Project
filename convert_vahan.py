#!/usr/bin/env python3
"""
Convert VAHAN "Vehicle Category x Month" exports into ev_registrations.csv.

Put one pair of files per state in a folder, named  <STATE>_ALL.*  and  <STATE>_EV.*
(e.g. UK_ALL.xlsx / UK_EV.xlsx, DL_ALL.csv / DL_EV.csv). The files may be real Excel
files or CSV text with an .xlsx name (VAHAN sometimes does this) - both are handled.
The state name is read from the title line inside each file.

Usage:  python convert_vahan.py <input_folder> [output_csv]
"""
import re, sys, glob, os, io
import pandas as pd

# VAHAN vehicle-category label -> pipeline category (2W/3W/4W/Bus/Other)
def map_category(label: str) -> str:
    s = label.upper()
    if s.startswith("TWO WHEELER"):
        return "2W"
    if s.startswith("THREE WHEELER"):
        return "3W"
    if s.startswith("FOUR WHEELER") or s in ("LIGHT MOTOR VEHICLE", "LIGHT PASSENGER VEHICLE"):
        return "4W"
    if s in ("HEAVY PASSENGER VEHICLE", "MEDIUM PASSENGER VEHICLE"):
        return "Bus"
    return "Other"   # goods vehicles, heavy/medium motor vehicle, other

MONTH_RE = re.compile(r"^(\d{4})-([A-Za-z]{3})$")

def read_text(path):
    """Return file content as text lines, whether it is real xlsx or CSV text."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":                       # genuine Excel file
        df = pd.read_excel(path, header=None, dtype=str)
        buf = io.StringIO(); df.to_csv(buf, index=False, header=False)
        return buf.getvalue().splitlines()
    with open(path, encoding="utf-8-sig") as f:
        return f.read().replace("\\n", "\n").splitlines()   # VAHAN writes a literal \n in the title

def parse(path):
    lines = read_text(path)
    title = lines[0].strip().strip('"')
    m = re.search(r"for (.+?) \(\d{4}\)", title)
    state = m.group(1).strip() if m else None
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Vehicle Category"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df = df[df["Vehicle Category"].str.strip().str.lower() != "total"]
    months = [c for c in df.columns if MONTH_RE.match(str(c))]
    long = df.melt(id_vars="Vehicle Category", value_vars=months, var_name="m", value_name="n")
    long["month"] = pd.to_datetime(long["m"], format="%Y-%b").dt.strftime("%Y-%m")
    long["vehicle_category"] = long["Vehicle Category"].map(map_category)
    long["n"] = pd.to_numeric(long["n"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    out = long.groupby(["month", "vehicle_category"], as_index=False)["n"].sum()
    return state, out

def main(folder, out_csv):
    rows, problems = [], []
    alls = sorted(glob.glob(os.path.join(folder, "*_ALL.*")))
    if not alls:
        sys.exit("No *_ALL.* files found in " + folder)
    for a in alls:
        e = re.sub(r"_ALL(\.[^.]+)$", r"_EV\1", a)
        if not os.path.exists(e):
            problems.append(f"missing EV file for {os.path.basename(a)}"); continue
        s_all, tot = parse(a); s_ev, ev = parse(e)
        if s_all != s_ev:
            problems.append(f"state mismatch {os.path.basename(a)}: '{s_all}' vs '{s_ev}'"); continue
        d = tot.rename(columns={"n": "total_registrations"}).merge(
            ev.rename(columns={"n": "ev_registrations"}), on=["month", "vehicle_category"], how="outer").fillna(0)
        d.insert(0, "state", s_all)
        bad = d[d.ev_registrations > d.total_registrations]
        if len(bad):
            problems.append(f"{s_all}: {len(bad)} rows have EV > total (check files)")
        rows.append(d)
    res = pd.concat(rows)[["state", "month", "vehicle_category", "ev_registrations", "total_registrations"]]
    res[["ev_registrations", "total_registrations"]] = res[["ev_registrations", "total_registrations"]].astype(int)
    res.sort_values(["state", "month", "vehicle_category"]).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {len(res)} rows, {res.state.nunique()} state(s)")
    for p in problems: print("WARNING:", p)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ev_registrations.csv")
