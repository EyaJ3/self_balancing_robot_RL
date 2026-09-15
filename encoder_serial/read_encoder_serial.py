"""Read encoder telemetry sent by the ESP32 over USB serial."""

from __future__ import annotations

import argparse
import csv
import serial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "port",
        help="Serial port, for example /dev/ttyUSB0 or /dev/serial0",
    )
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    with serial.Serial(args.port, args.baud, timeout=1) as device:
        reader = csv.reader(
            line.decode("ascii", errors="replace").strip()
            for line in device
        )
        for row in reader:
            if len(row) != 6 or row[0] != "ENC":
                continue
            _, millis, left_count, right_count, left_rad_s, right_rad_s = row
            print(
                f"t={millis} ms | "
                f"L count={left_count} rad/s={left_rad_s} | "
                f"R count={right_count} rad/s={right_rad_s}",
                flush=True,
            )


if __name__ == "__main__":
    main()
