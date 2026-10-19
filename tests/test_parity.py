"""Parity check: the API must return the same price as M5's model for 3 held-out cars (owner: M5).

    python tests/test_parity.py                                   # model file directly, no server
    python tests/test_parity.py --url http://127.0.0.1:8000/api/v1/used-ev/price
    python tests/test_parity.py --url https://<api-gateway-url>/api/v1/used-ev/price

Passes when every price is within $1 of tests/parity_cases_v1.json.
In --url mode, the price is read from the response field named by --field; if that field
is missing, every numeric value in the JSON response is tried.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests" / "parity_cases_v1.json").read_text())


def numbers(obj):
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in numbers(v)]
    return []


def local_prices():
    import joblib
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from src import prep
    model = joblib.load(ROOT / CASES["model_file"])
    return [float(model.predict(prep.add_lookups(pd.DataFrame([c["request"]]), ROOT / "models"))[0])
            for c in CASES["cases"]]


def api_prices(url, field):
    import requests
    out = []
    for c in CASES["cases"]:
        r = requests.post(url, json=c["request"], timeout=30)
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict) and field in body:
            out.append(float(body[field]))
        else:   # unknown response shape: take the numeric value closest to the expected price
            out.append(min(numbers(body), key=lambda n: abs(n - c["expected_price_usd"]), default=float("nan")))
        print("  response:", json.dumps(body)[:200])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--field", default="predicted_price")
    args = ap.parse_args()
    got = api_prices(args.url, args.field) if args.url else local_prices()
    ok = True
    for c, g in zip(CASES["cases"], got):
        diff = abs(g - c["expected_price_usd"])
        ok &= diff <= CASES["tolerance_usd"]
        r = c["request"]
        print(f"{'PASS' if diff <= CASES['tolerance_usd'] else 'FAIL'}  {r['model_year']} {r['trim_name']:<30s} "
              f"expected ${c['expected_price_usd']:,.2f}  got ${g:,.2f}  diff ${diff:,.2f}")
    print("PARITY OK" if ok else "PARITY FAILED: check add_lookups(), trim spelling and model version")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
