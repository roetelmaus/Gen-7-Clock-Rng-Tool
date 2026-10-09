# Gen 7 Clock RNG Tool

A Windows desktop tool that recognizes Gen 7 clock positions from a selected
game, emulator, OBS projector, or webcam preview window and shows the starting clock hands.

Made as a helper for **3DSRNGTool**. In **Gen 7 Main RNG Tool**, select
**Start Position** when entering the results.

## Download

[Download the latest version](https://github.com/roetelmaus/Gen-7-Clock-Rng-Tool/releases/latest)

Download **Gen.7.Clock.RNG.V2.1.exe** and open it. Windows is required.
No Python installation, template folder, or additional files are needed.

## Getting Started

1. Open the tool and select the window showing your game.
2. Click **Set Clock Box** and draw a box around the area where the clock appears.
3. Go back with **B**, then press **A** to show the clock again. Wait for calibration.
4. **Restart the game after calibration.** The calibration clock is not counted.
5. Click **Restart** in this tool before showing the first clock after the game restarts.
6. Collect eight clocks and copy the comma-separated sequence into 3DSRNGTool.

Keep the window and game image size unchanged after calibration. Do not skip a
clock: if one is missed, restart the game and begin a new sequence.

## Using a Real 3DS and Webcam

1. Select your webcam preview window, for example **Camo Studio**.
2. Enable **Webcam Mode** and click **Set Screen Corners**.
3. Click the four corners of the **upper display**, not the console body:
   **top-left, top-right, bottom-right, bottom-left**.
4. Continue with **Set Clock Box** and the steps above.

Keep the whole upper display visible, with good lighting and a sharp image.
The clock box follows small movements automatically, including menu transitions.
Large movements or changes in camera angle may require a new calibration.

If **SCREEN TRACKING LOST** appears, bring all four display corners back into
view. Recognition resumes when the display is found again, and the collected
numbers stay on screen. If a clock appeared while tracking was lost, restart the
game and the sequence.

## Controls

- **Last Number** shows the latest result and its clock picture.
- The sequence stores the first **eight numbers**. Later clocks still update Last Number.
- **Restart** clears the sequence but keeps the current calibration.
- **Stop Searching** stops capture and turns the preview black. Press Restart to resume.
- After **two minutes without clock activity**, press Restart before showing the next clock.
- Selecting another window or changing Webcam Mode requires a new calibration.

## Troubleshooting

- **Calibration fails:** make the game image larger or smaller and try Set Clock Box again.
- **Wrong or missing numbers:** check focus, reflections, lighting, and image size. Recalibrate
  if the picture changed. A successful match is not a guarantee of a correct sequence.
- **No seed found in 3DSRNGTool:** check Start Position, the selected game version, and
  that the sequence starts at the first clock after restarting the game, without gaps.

## About

[What's new](CHANGELOG.md) | [Developer documentation](DEVELOPMENT.md) | [GPL-3.0 license](LICENSE)

The original source code is licensed under GPL-3.0-only. Third-party dependencies
and Pokemon game imagery retain their respective rights. This unofficial tool
is not affiliated with Nintendo, The Pokemon Company, or Game Freak.
