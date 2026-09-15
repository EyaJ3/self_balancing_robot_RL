"""Measure the real motor start threshold with the wheels lifted.

Teste chaque niveau de PWM d'abord en marche avant, puis en marche arrière,
pour tous les moteurs sélectionnés.
"""

from __future__ import annotations

import argparse
import time

import RPi.GPIO as GPIO


RPWM_RIGHT, LPWM_RIGHT, REN_RIGHT, LEN_RIGHT = 12, 13, 24, 25
RPWM_LEFT, LPWM_LEFT, REN_LEFT, LEN_LEFT = 19, 18, 26, 27


def run_sweep(
    direction: str,
    channels: dict[str, "GPIO.PWM"],
    motor: str,
    levels: list[float],
    hold: float,
) -> None:
    """Run one sweep of PWM levels in the given direction ('forward' or 'reverse')."""
    print(f"--- Marche {direction} ---", flush=True)
    for level in levels:
        if not 0 < level <= 100:
            raise ValueError("Each PWM level must be between 0 and 100")
        for pwm in channels.values():
            pwm.ChangeDutyCycle(0)
        if motor in ("both", "right"):
            channels[f"right_{direction_key(direction)}"].ChangeDutyCycle(level)
        if motor in ("both", "left"):
            channels[f"left_{direction_key(direction)}"].ChangeDutyCycle(level)
        print(f"[{direction}] PWM={level:.1f}% for {hold:.1f}s", flush=True)
        time.sleep(hold)
        for pwm in channels.values():
            pwm.ChangeDutyCycle(0)
        time.sleep(0.5)


def direction_key(direction: str) -> str:
    return "forward" if direction == "avant" else "reverse"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motor",
        choices=("both", "left", "right"),
        default="both",
        help="Motor to test; wheels must be lifted.",
    )
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[5, 8, 10, 12, 15, 20, 25, 30],
    )
    args = parser.parse_args()
    if args.hold <= 0:
        raise ValueError("--hold must be positive")

    motor_pins = [
        RPWM_RIGHT,
        LPWM_RIGHT,
        REN_RIGHT,
        LEN_RIGHT,
        RPWM_LEFT,
        LPWM_LEFT,
        REN_LEFT,
        LEN_LEFT,
    ]
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    pwms: list[object] = []
    try:
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        for pin in [REN_RIGHT, LEN_RIGHT, REN_LEFT, LEN_LEFT]:
            GPIO.output(pin, GPIO.HIGH)

        channels = {
            "right_forward": GPIO.PWM(RPWM_RIGHT, 1000),
            "right_reverse": GPIO.PWM(LPWM_RIGHT, 1000),
            "left_forward": GPIO.PWM(LPWM_LEFT, 1000),
            "left_reverse": GPIO.PWM(RPWM_LEFT, 1000),
        }
        pwms = list(channels.values())
        for pwm in pwms:
            pwm.start(0)

        print("Wheels must be completely off the ground.")
        print("Observe whether each selected wheel starts rotating at each level.")
        input("Press Enter to start, or Ctrl+C to abort...")

        # Marche avant
        run_sweep("avant", channels, args.motor, args.levels, args.hold)

        # Petite pause de sécurité entre les deux sens
        time.sleep(1.0)

        # Marche arrière
        run_sweep("arriere", channels, args.motor, args.levels, args.hold)

        print("Test complete.")
    finally:
        for pwm in pwms:
            try:
                pwm.ChangeDutyCycle(0)
                pwm.stop()
            except (AttributeError, TypeError):
                pass
        GPIO.output([REN_RIGHT, LEN_RIGHT, REN_LEFT, LEN_LEFT], GPIO.LOW)
        GPIO.cleanup()


if __name__ == "__main__":
    main()