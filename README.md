# Two-wheel robot reinforcement learning

This project trains the robot in `self_balancing_robot.urdf` with PPO and
PyBullet. The policy outputs two independent actions, one torque command per
wheel, so it can compensate for the two motors' different deadbands.

Training uses moderate domain randomization: link masses, wheel friction,
motor torque, motor deadband and delay, sensor noise/bias, encoder noise and
delay, and initial tilt vary between episodes. Nominal (non-randomized) mode
uses a fixed realistic deadband (left 0.02, right 0.085) instead of zero.

## Install

From this folder:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Check the environment

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --check-env --timesteps 1
```

## Train

Start with a longer run after the check succeeds:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --timesteps 500000
```

The model is saved as `training_output/ppo_balance_robot.zip` and its
observation-normalization statistics are saved as
`training_output/vecnormalize.pkl`. Keep both files together.

## Train the encoder-free hardware policy

Because the wheel encoders are unreliable, the Raspberry Pi policy uses only
the MPU and the filtered motor command:

```text
[angle error, gyro rate, filtered motor command]
```

Train it separately so the existing six-observation simulation policy is not
overwritten:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --hardware --timesteps 2000000 --envs 8
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe export_hardware_policy.py
```

To test whether randomization is making learning too difficult, train a
separate nominal hardware policy:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --hardware --no-randomization --timesteps 2000000 --envs 8
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe export_hardware_policy.py --nominal
```

View it with:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe run_trained_balance.py --hardware --nominal
```

The nominal export can be selected on the Pi with `--nominal`.

## Filtered acceleration experiment

This cautious experiment adds one fourth observation: gravity-corrected,
low-pass-filtered forward acceleration. It does not integrate acceleration, so
it cannot accumulate velocity drift:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --hardware-accel --no-randomization --timesteps 2000000 --envs 8
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe run_trained_balance.py --hardware-accel --nominal
```

Export it for the Raspberry Pi only after it passes simulation:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe export_hardware_policy.py --accel --nominal
```

The full-observation baseline can be trained independently:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --full-baseline --timesteps 2000000 --envs 8
```

This creates `training_output_full_baseline/ppo_balance_robot.zip` and
`training_output_full_baseline/vecnormalize.pkl`. View it with:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe run_trained_balance.py --full-baseline
```

For a short frame-stacking experiment, use four recent hardware frames
(`12` observation values total) in a separate output directory:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --hardware-stacked --timesteps 400000 --envs 8
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe run_trained_balance.py --hardware-stacked
```

The ESP32 encoder telemetry applies the calibrated left/right signs and sends
wheel angular velocity in rad/s using `3948` quadrature counts per revolution.
The Raspberry Pi reader expects:

```text
ENC,millis,left_count,right_count,left_rad_s,right_rad_s
```

For runtime wiring without USB, the ESP32 firmware uses UART2:

```text
ESP32 GPIO33 (TX) -> Raspberry Pi GPIO15 / RXD (physical pin 10)
ESP32 GPIO32 (RX) <- Raspberry Pi GPIO14 / TXD (physical pin 8)
ESP32 GND         -> Raspberry Pi GND (physical pin 6)
```

Both devices use 3.3 V logic. Do not connect 5 V to either UART signal.
After enabling the Pi hardware UART and disabling its serial console, read it
with:

```bash
python3 read_encoder_serial.py /dev/serial0
```

The next RL experiment uses the matching five-value observation:

```text
[pitch, gyro rate, left wheel speed, right wheel speed, filtered command]
```

Train and view it separately:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe train_balance.py --hardware-encoder --timesteps 2000000 --envs 8
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe run_trained_balance.py --hardware-encoder
```

Export only after the simulation policy is stable:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe export_hardware_policy.py --encoder
```

On the Raspberry Pi, the matching deployment command is sensor-only first:

```bash
python3 deploy_pi_hardware.py --encoder --encoder-port /dev/ttyUSB0
```

The hardware runner normalizes observations with the saved statistics and is
sensor-only by default:

```bash
python3 deploy_pi_hardware.py
```

Only enable motors after the safety checks, with the wheels lifted:

```bash
python3 deploy_pi_hardware.py --arm --max-pwm 20
```

The runner uses the measured `1.6` degree upright offset, a 120 Hz hardware
loop, a stable 240 Hz PyBullet physics timestep with two substeps per policy
action, a 25-degree tilt cutoff, immediate shutdown on sensor failure, and
always stops the motors during exit.

## Hardware bring-up notes (encoder policy `_v3`)

State reached on 2026-09-20. These settings were found by testing on the real
robot; do not undo them without re-testing.

### Policy and files

`training_output_hardware_encoder_v3/best` was trained with:

```powershell
python train_balance.py --hardware-encoder --timesteps 2000000 --envs 8 --episode-seconds 20 --lr-decay --velocity-penalty 0.2 --position-penalty 0.1 --output-suffix _v3
python export_hardware_policy.py --encoder --output-suffix _v3 --best
python verify_export.py --encoder --output-suffix _v3 --best
```

Always use the `best/` checkpoint. The final training step of `_v3` collapsed,
so the `ppo_balance_robot.zip` and `vecnormalize.pkl` in the folder root must
not be exported. `verify_export.py` must print `Match: True` on every line.

Copy only `deploy_pi_hardware.py` and
`training_output_hardware_encoder_v3/hardware_policy.npz` to the Pi (it holds
the weights and the normalization statistics). `deploy_pi_hardware.py` needs
`numpy`, `pyserial`, `smbus2` and `RPi.GPIO` (see `requirements-pi.txt`). The
script defaults to the old `training_output_hardware_encoder` policy, so pass
`--policy` explicitly, and check the file with `md5sum` on the Pi.

### Sign conventions (verified on the robot)

The policy expects: tilt toward the side where the IMU angle is positive
(side P) gives a positive action, positive wheel speed, and wheels rolling
toward P.

- Encoders: `encoder_serial.ino` uses `LEFT_ENCODER_SIGN = -1.0f` and
  `RIGHT_ENCODER_SIGN = 1.0f`. Both wheels read positive `rad/s` when the robot
  rolls toward P. The Pi applies no encoder sign of its own; do not add one.
- Pitch gyro: the angle is `atan2(ax, az)`, which has the opposite sign to the
  MPU6050 Y gyro. Run with `--invert-gyro` so that the gyro equals d(angle)/dt,
  as in simulation. Without it the complementary filter fights itself and the
  policy gets anti-damping: the robot swings +-15 degrees and saturates its
  action for 70-90 % of the time. Integrated gyro matches the accelerometer
  angle change within about 6 %, so the gyro scale (131 LSB per deg/s) is right.
- Motor direction: with the wheels lifted, tilting toward P makes both wheels
  turn toward P.

### Working command

```bash
python3 deploy_pi_hardware.py --encoder --encoder-port /dev/ttyS0 --policy /home/pi/robot_bul/training_output_hardware_encoder_v3/hardware_policy.npz --arm --max-pwm 40 --invert-gyro --seconds 35
```

Before arming: confirm `python3 read_encoder_serial.py /dev/ttyS0` prints
`ENC` lines with both wheel speeds positive when pushing toward P, keep the
wheels lifted for the first armed run, and keep an emergency power disconnect
within reach.

### Results

Free standing (no hands) time, by `--max-pwm`, as observed on the robot:

| `--max-pwm` | free standing time |
|---|---|
| 20 | under 2 s |
| 30 | up to about 15 s |
| 40 | at least 35 s (longer not tested) |
| 50 | at least 35 s (longer not tested) |

Compared on one 35 s log each (after the first 6 s, the robot's initial move):

| | 40 % | 50 % |
|---|---|---|
| mean absolute tilt | 2.15 deg | 2.43 deg |
| wobble | 2.1 Hz, +-2.9 deg | 2.5 Hz, +-3.3 deg |
| mean absolute gyro | 30 deg/s | 38 deg/s |
| steps at full action, left / right | 6 % / 24 % | 9 % / 30 % |
| forward creep after 6 s | about 1.0 cm/s (0.26 m in 29 s) | about 0.1 cm/s (0.01 m in 29 s) |

At 50 % the robot stays in place but wobbles a little more; at 40 % it wobbles
less but slowly creeps forward. Both runs also roll about 0.3 m forward in the
first seconds after release. Each figure comes from a single run, so repeat
before treating the creep difference as certain.

The 35 s run at 40 (`free_40pwm_35s.csv`): tilt never passed +-5 degrees (mean
absolute 2.15 degrees), 6 % / 24 % of steps at full action on the left / right
wheel, a steady 2.1 Hz wobble of about +-3 degrees that does not grow, slow
creep of about 0.35 m and about 15 degrees of heading change, loop at 119 Hz.
The right wheel saturates more than the left because of its larger deadband.

### Run logs

Every run writes `logs/deploy_<date>_<time>.csv` (change with `--log`, disable
with `--no-log`). It is written when the run ends, and `logs/` is git-ignored.
Columns: `t_s`, `dt_s`, `angle_deg` (filtered), `error_deg` (filtered angle
minus 1.6), `gyro_deg_s`, `left_rad_s`, `right_rad_s`, `action_left`,
`action_right`, `pwm_left_pct`, `pwm_right_pct`, `accel_angle_deg` (raw
accelerometer angle). Copy the logs to the PC with:

```powershell
scp "pi@<pi-address>:/home/pi/robot_bul/logs/*.csv" C:\Users\USER\Desktop\robot_bul\logs\
```

A healthy run has nonzero `left_rad_s` / `right_rad_s` in almost every row.
All zeros mean the encoder link is dead.

### Known limitations

- The script does not detect stale encoder data. If the link dies it keeps
  the last speeds (zeros at start) and runs blind on two of six inputs. Check
  the log for zero wheel speeds after every run.
- It needs about 40 % PWM to stand: the simulation's full action (0.315 N*m)
  does not match the real motors, so the real loop is torque-limited at low
  `--max-pwm`. The simulation's deadband is applied to the action, while the
  measured deadband is a fraction of the PWM range, so the two agree only at
  `--max-pwm 100`.
- Battery charge matters: a sagging supply behaves like a lower `--max-pwm`.
- The 2.1 Hz wobble is the remaining sim-to-real gap (likely deadband and
  delay).

## Compare policies headlessly

`evaluate_policy.py` reports fall rate, survival time and drift over many
seeds. Run it on a policy before deploying and compare against the old one:

```powershell
python evaluate_policy.py --hardware-encoder
python evaluate_policy.py --hardware-encoder --nominal
python evaluate_policy.py --hardware-encoder --output-suffix _v2 --best
```

## Retrain without overwriting an existing model

`--output-suffix` writes to a new folder. Training keeps evaluating on the
randomized environment and saves the best checkpoint (with matching
normalization stats) in `best/`:

```powershell
python train_balance.py --hardware-encoder --timesteps 2000000 --envs 8 --episode-seconds 20 --lr-decay --output-suffix _v2
python export_hardware_policy.py --encoder --output-suffix _v2 --best
python verify_export.py --encoder --output-suffix _v2 --best
```

Always run `verify_export.py` before trusting or deploying a new `.npz`.
`export_hardware_policy.py` and `verify_export.py` share the same folder for
`hardware_policy.npz`.

## Watch the trained robot

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe run_trained_balance.py
```

To view the separate encoder-free hardware policy in PyBullet:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe run_trained_balance.py --hardware
```

Training is initially done headlessly, so the GUI is only opened when running
the saved policy. Use TensorBoard if you want to inspect training:

```powershell
C:\Users\USER\Desktop\robot_bul\.venv\Scripts\python.exe -m tensorboard.main --logdir training_output/tensorboard
```
