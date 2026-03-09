# PhysicsLab Pro

PhysicsLab Pro is a PyQt6 desktop application for optics experiments and basic lab-data processing. The current codebase is centered on a small package plus a lightweight root entrypoint:

- `main.py`: lightweight startup entrypoint
- `physicslab/`: modularized GUI package
- `algorithms.py`: image-analysis, peak detection, fitting, uncertainty, and statistics utilities

## Current Functional Scope

The application currently includes:

- Optics image analysis: load an interference image, pick two points, extract line intensity, detect peaks, and plot the signal
- Optics video analysis: load a video, pick a center point, track brightness over frames, smooth the signal, and estimate peak counts
- Virtual simulation: Newton's rings, wedge interference, and double-slit interference rendered with OpenGL
- Data workstation: table-based input, CSV import, descriptive statistics, fitting, uncertainty calculation, and chart rendering
- Vibration lab: simple harmonic motion simulation with spring-mass animation, x/t-v/t-a/t curves, and a rotating phasor view
- AI assistant dock: sends questions to DeepSeek through the OpenAI-compatible SDK

## Project Structure

```text
physicslab-pro/
├─ physicslab/
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ app.py
│  ├─ ai_assistant.py
│  ├─ data_workstation.py
│  ├─ optics_tab.py
│  ├─ simulation.py
│  ├─ vibration_tab.py
│  ├─ widgets.py
│  └─ workers.py
├─ main.py
├─ algorithms.py
├─ requirements.txt
├─ README.md
└─ 牛顿环.png
```

## Environment

Validated locally with:

- Python `3.10.11`
- PyQt6 importable
- OpenCV / NumPy / SciPy / Matplotlib importable
- OpenGL bindings importable
- `pandas` installed
- `main.py` and `algorithms.py` passing `py_compile`
- `MainWindow` instantiating successfully with `QT_QPA_PLATFORM=offscreen`

## Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Configure the DeepSeek key if you want to use the AI assistant:

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_key"
```

Run the app:

```powershell
python main.py
```

You can also run the package entrypoint:

```powershell
python -m physicslab
```

## Architecture Notes

- The root entrypoint is now thin; the GUI code has been split into package modules for widgets, workers, tabs, simulation, AI dock, and app bootstrap.
- `VirtualLabTab` exists in the codebase but is not attached to the main window at startup.
- `pandas` is only required when CSV import/export features are used, but it is a runtime dependency for those paths.
- The AI assistant uses the OpenAI Python SDK against DeepSeek's compatible API endpoint.

## Known Risks

- There is no automated test suite yet.
- There is no packaging or installer workflow yet.
- Some modules are still large, especially the two main tabs, so the next refactor step should be extracting controller/service logic from widget classes.
