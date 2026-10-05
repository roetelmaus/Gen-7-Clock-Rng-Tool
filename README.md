# Gen 7 Clock RNG Tool

A Windows desktop tool that recognizes Gen 7 clock positions from a selected
game, emulator, or OBS projector window and shows the starting clock hands.

This tool is intended as a helper for **3DSRNGTool**. In 3DSRNGTool's
**Gen 7 Main RNG Tool**, select **Start Position** when using the results.

## Download

Download **Gen 7 Clock RNG V2.exe** from the
[latest release](https://github.com/roetelmaus/Gen-7-Clock-Rng-Tool/releases/latest).
The executable includes Python, all dependencies, and clock templates.
No installation or additional files are required.

## Features

- Select a visible Windows window as the capture source.
- Calibrate clock position and scale with **Set Clock Box**.
- Match embedded templates and quality variants with small pixel-offset correction.
- View the latest number alongside its matching clock template.
- Collect eight values in a copyable, comma-separated sequence.
- Continue updating the latest number after the sequence is full.
- Stop capture with **Stop Searching** and resume with **Restart**.
- Show only the clock ROI after calibration, with a preview limited to 10 FPS.
- Pause detection after two minutes without clock activity.

## Usage

1. Launch the tool and select the source window.
2. Click **Set Clock Box** and draw a rough box where the clock appears.
3. Go back with **B**, then press **A** to show the clock again.
4. After successful calibration, detection starts automatically. The calibration
   clock is excluded from the results.
5. Copy the sequence, or click **Restart** to clear it and start a new sequence.

**Stop Searching** turns the preview black and stops capture. Restart preserves
the calibrated ROI. Changing the source window requires a new calibration.
After the two-minute idle timeout, press **Restart before showing the next clock**.

## Recognition and Settings

Results require a similarity score of at least **0.900**, with a **0.020** margin
over the runner-up class. The output applies a **+4 offset modulo 17** to the
detected template number. V2 does not subtract a stage after waiting for a match.

Templates include blur and downsampling variants. Severe loss of hand detail,
fade timing, and source resolution can still affect recognition. Similarity
scores are not probabilities or guarantees of correctness.

The executable stores settings in `%LOCALAPPDATA%\Gen7ClockRNG_V2\config.json`.
Set `show_score_debug` to `true` there to display the matching score.
V2 uses settings separate from the original version.

## Run from Source

Windows and Python 3.9 or newer are required.

```powershell
python -m pip install -r requirements.txt
python obs_clock_gui.py
```

Or double-click `start_gui.bat` after installing the dependencies.
Source runs use the local `config.json`.

## Build the Standalone EXE

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --onefile --windowed --name "Gen 7 Clock RNG V2" --add-data "config.json;." --add-data "templates;templates" obs_clock_gui.py
```

The executable is created in `dist/`.

## Validation

```powershell
python test_quality.py
python test_gui_v2.py
python test_roi_performance.py
```

The synthetic comparison covers 170 template-derived inputs with shifts,
downsampling, and blur. GUI tests check stop/restart and preview behavior.
The ROI test checks pixel equality with full-frame alignment and measures
transformation time. These checks do not replace independent gameplay testing.

## Template Tools

- `list_windows.py`: list visible capture windows.
- `calibrate_roi.py`: manually save an ROI for source-based helper workflows.
- `save_template_from_roi.py`: save a numbered template from the configured ROI.
- `record_fade_templates.py`: capture additional template variants.

Canonical templates are named `0.png` through `16.png`. Additional variants
use the same leading number, for example `2_fade_example.png`.
