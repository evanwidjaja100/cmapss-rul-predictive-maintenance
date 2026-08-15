"""Serve the frozen RUL model: HTTP endpoint or batch mode.

Usage:
    .venv/Scripts/python.exe scripts/serve.py serve --port 8000
    .venv/Scripts/python.exe scripts/serve.py batch --input data/raw/test_FD001.txt --rul data/raw/RUL_FD001.txt
    .venv/Scripts/python.exe scripts/serve.py batch --input data/raw/test_FD001.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from rul_prediction.serving.inference import RulPredictor

ROOT = Path(__file__).resolve().parents[1]


def _load_rul(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, sep=r"\s+", engine="python")[0].to_numpy(dtype=int)


def batch(args: argparse.Namespace, predictor: RulPredictor) -> None:
    result = predictor.predict_file(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["unit_id", "n_cycles", "padded_short", "prediction"])
        for i in range(len(result["unit_id"])):
            writer.writerow([result["unit_id"][i], result["n_cycles"][i],
                             result["padded_short"][i], result["prediction"][i]])
    print(f"wrote {len(result['unit_id'])} predictions -> {out}")
    if args.rul:
        from rul_prediction.evaluation.metrics import mae, r2, rmse
        from rul_prediction.evaluation.nasa_score import nasa_score

        true = np.minimum(_load_rul(Path(args.rul)), predictor.max_rul)
        pred = np.asarray(result["prediction"], dtype=float)
        print(f"RMSE={rmse(true, pred):.4f}  MAE={mae(true, pred):.4f}  "
              f"R2={r2(true, pred):.4f}  NASA={nasa_score(true, pred):.3f}")


class _Handler(BaseHTTPRequestHandler):
    predictor: RulPredictor = None  # set by serve()

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": "xgboost",
                             "variant": self.predictor.variant,
                             "max_rul": self.predictor.max_rul})
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):  # noqa: N802
        if self.path != "/predict":
            return self._send(404, {"error": f"unknown path {self.path}"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            frame = pd.DataFrame(payload["rows"])
            result = self.predictor.predict(frame)
            self._send(200, result)
        except Exception as exc:  # noqa: BLE001 - report malformed input
            self._send(400, {"error": str(exc)})

    def log_message(self, fmt, *args):  # quiet request logs
        pass


def serve(args: argparse.Namespace, predictor: RulPredictor) -> None:
    _Handler.predictor = predictor
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"RUL service on http://127.0.0.1:{args.port}  (GET /health, POST /predict)")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen RUL model service")
    parser.add_argument("--config", default=str(ROOT / "configs" / "final_model.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=8000)

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--input", required=True, help="C-MAPSS test-style text file")
    p_batch.add_argument("--rul", help="optional RUL_FD001.txt for metrics")
    p_batch.add_argument("--output", default="experiments/serving_predictions.csv")

    args = parser.parse_args()
    predictor = RulPredictor(args.config)
    if args.command == "serve":
        serve(args, predictor)
    else:
        batch(args, predictor)


if __name__ == "__main__":
    main()