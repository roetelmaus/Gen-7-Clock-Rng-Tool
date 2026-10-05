@echo off
cd /d "%~dp0"
python obs_clock_gui.py
if errorlevel 1 pause
