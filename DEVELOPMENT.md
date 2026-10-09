# Development

## Run from Source

Windows and Python 3.9 or newer are required.

```powershell
python -m pip install -r requirements.txt
python obs_clock_gui.py
```

Source runs use the local config.json. Packaged runs store user settings in
`%LOCALAPPDATA%\Gen7ClockRNG_V2\config.json`. Set `show_score_debug` to true
to display the score. V2 settings are separate from the original version.

## Build

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm "Gen 7 Clock RNG V2.1.spec"
```

The single-file executable is created in dist/. The spec bundles templates,
default configuration, and the GPL license. Private camera diagnostics are not
part of the build. Distribute corresponding source with GPL-licensed binaries.

## Tests

```powershell
python test_quality.py
python test_gui_v2.py
python test_roi_performance.py
python -m unittest test_screen_tracker
```

Template tests cover 170 synthetic inputs. GUI tests cover restart, warning,
and preview behavior. ROI tests check pixel equality with full-frame alignment.
Tracking tests cover menu changes, black transitions, motion, and immediate
reacquisition. The optional private Camo-image test is skipped when absent.
These tests do not establish a gameplay success rate.

## Matching and Tracking

Matches require score >= 0.900 and margin >= 0.020. Output uses +4 modulo 17.
The GUI does not subtract a stage after waiting. Quality variants include blur
and downsampling; scores are similarities, not probabilities.

Webcam tracking uses shape-constrained display contours at 5 Hz, preserving
the clicked corner offsets. Tracking loss triggers a retry on every captured
frame. Normal loss warns after two seconds; recognized black transitions have
a ten-second warning grace period without delaying recovery. Two dark frames
separate clock appearances. Screen-corner calibration is session-only.

## Template Utilities

- list_windows.py lists visible capture windows.
- calibrate_roi.py saves an ROI for source-based helper workflows.
- save_template_from_roi.py saves a numbered template from the configured ROI.
- record_fade_templates.py captures additional template variants.

Canonical templates are 0.png through 16.png. Variants retain the leading number,
for example 2_fade_example.png. Legacy comparison settings remain for helper tools;
the GUI uses QualityMatcher.
