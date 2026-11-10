# YOLO11 vs YOLO26 Benchmark for Pea-Plant Detection

Trains, validates and benchmarks four detectors (yolo11n, yolo11s, yolo26n, yolo26s)
on a pea-plant dataset, logs accuracy / latency / power / energy to CSV, and exports
weights (.pt and .onnx) for a Jetson Orin Nano.

## Reference hardware
- CPU: Intel Core Ultra 9 185H
- GPU: NVIDIA RTX 4070 Laptop (8 GB VRAM)
- RAM: 64 GB
- OS / NVIDIA driver / CUDA: <fill in>

## Setup
```bash
git clone <repository URL>
cd pea-yolo-benchmark
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python version used: <fill in>. `requirements.txt` pins the exact versions of
torch, ultralytics, opencv-python and pynvml used in the experiment.

## Data
The dataset is not stored in this repository. Obtain it from <dataset source>, then:
1. Edit the paths in `pea_plants.yaml` to point to your train / val / test folders.
2. Place one representative image at `pea_sample.jpg` in the repository root
   (used for the latency benchmark).

## Run
```bash
python benchmark.py
```
For each model the script:
1. trains for up to 100 epochs (patience 25) at imgsz 1280, rect=True, AMP on;
   batch 8 for the n models and 6 for the s models;
2. validates on the `test` split (falls back to `val` if absent);
3. benchmarks inference at 960x1280, FP16: 30 warm-up frames, then 300 timed frames,
   while sampling GPU power and temperature via NVML every 0.1 s;
4. exports `best.pt` and ONNX (opset 12, simplified) to `exported_for_jetson/`.

## Outputs
- `desktop_thesis_benchmark.csv`: one row per model (mAP50, mAP50-95, precision,
  recall, latency, FPS, average / peak power, J/frame, training time and energy,
  peak GPU temperature)
- `runs/detect/thesis_runs/<model>_pea_1280/`: Ultralytics training logs and weights
- `exported_for_jetson/`: `.pt` and `.onnx` weights

## Resuming after a crash
The script can be re-run safely:
- models already in the CSV are skipped;
- if `last.pt` exists, training resumes;
- if `best.pt` exists, training is skipped. In that case training power is NOT
  measured: it is estimated at 95 W, and training time is read from `results.csv`.

For fully measured results, delete `runs/`, `exported_for_jetson/` and the CSV,
and run once without interruption.

## Tests
```bash
pip install pytest
pytest -v
```

## Expected variability
Accuracy varies slightly between runs (cuDNN benchmark mode and TF32 are enabled).
Latency, power and temperature depend on hardware, driver, ambient temperature and
laptop power profile, so compare trends rather than absolute values on other machines.
