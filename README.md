HEAD
# Two-wheel robot reinforcement learning

This project trains the robot in `self_balancing_robot.urdf` with PPO and
PyBullet. The policy controls one shared forward/backward torque for both
wheels.

Training uses moderate domain randomization: link masses, wheel friction,
motor torque, and initial tilt vary between episodes. Evaluation remains
nominal so the saved policy can be compared consistently.

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
