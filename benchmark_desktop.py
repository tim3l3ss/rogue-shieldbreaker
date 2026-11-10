import csv
import gc
import os
import shutil
import threading
import time
from pathlib import Path

# 1. Prevent 8GB VRAM fragmentation across sequential 1280x960 runs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import pynvml
import torch
from ultralytics import YOLO

# ==============================================================================
# HARDWARE OPTIMIZATIONS FOR:
#   CPU: Intel Core Ultra 9 185H | GPU: RTX 4070 Laptop 8GB | RAM: 64 GB
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

cv2.setNumThreads(0)
torch.set_num_threads(6)

# ==============================================================================
# EXPERIMENT CONFIGURATION (1320x989 -> 960x1280 4:3 multiple of 32)
# ==============================================================================
DATA_YAML = "pea_plants.yaml"
SAMPLE_IMAGE = "pea_sample.jpg"
MODELS = [
    "yolo11n", "yolo11s", "yolo11m",
    "yolo26n", "yolo26s", "yolo26m",
]

BATCH_MAP = {
    "yolo11n": 8,
    "yolo26n": 8,
    "yolo11s": 6,
    "yolo26s": 6,
    "yolo11m": 3,   # Safe for RTX 4070 8GB VRAM at imgsz=1280
    "yolo26m": 4,   # Safe for RTX 4070 8GB VRAM at imgsz=1280
}

# Estimated average RTX 4070 training power (W) used ONLY if a model was already
# trained in a previous session before the script crashed (e.g., yolo11n):
FALLBACK_TRAIN_WATTS = 95.0
FALLBACK_YOLO11N_HOURS = 0.890         # From your terminal log: "100 epochs completed in 0.890 hours"

TRAIN_IMGSZ = 1280
INFER_IMGSZ = (960, 1280)
EPOCHS = 100
PATIENCE = 25
DATALOADER_WORKERS = 6
CACHE_MODE = "ram"
EVAL_SPLIT = "test"                    # Automatically falls back to "val" if "test" is not in YAML

DEVICE = 0
WARMUP_FRAMES = 30
BENCHMARK_FRAMES = 300
COOLDOWN_TARGET_TEMP_C = 60
OUTPUT_CSV = Path("desktop_thesis_benchmark.csv")
EXPORT_DIR = Path("exported_for_jetson")


class LaptopGPUMonitor:
    """Samples RTX 4070 Laptop GPU power (W) and temperature (C) via NVML."""

    def __init__(self, gpu_index=0, interval_sec=0.1):
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.samples_watts = []
        self.samples_temp_c = []
        self._running = False
        self._thread = None
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.gpu_name, bytes):
            self.gpu_name = self.gpu_name.decode("utf-8")

    def get_current_temp(self):
        try:
            return pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            return 0

    def _poll(self):
        while self._running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                temp_c = pynvml.nvmlDeviceGetTemperature(
                    self.handle, pynvml.NVML_TEMPERATURE_GPU
                )
                self.samples_watts.append(power_mw / 1000.0)
                self.samples_temp_c.append(temp_c)
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval_sec)

    def start(self):
        self.samples_watts = []
        self.samples_temp_c = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        if not self.samples_watts:
            return 0.0, 0.0, 0.0
        avg_w = sum(self.samples_watts) / len(self.samples_watts)
        peak_w = max(self.samples_watts)
        peak_temp = max(self.samples_temp_c) if self.samples_temp_c else 0.0
        return avg_w, peak_w, peak_temp

    def wait_for_cooldown(self, target_temp_c=60, max_wait_sec=45):
        t0 = time.time()
        while (time.time() - t0) < max_wait_sec:
            temp = self.get_current_temp()
            if temp <= target_temp_c:
                break
            print(
                f"  [Thermal Guard] Cooling RTX 4070 ({temp}°C -> {target_temp_c}°C)...",
                end="\r",
            )
            time.sleep(3)
        print(" " * 80, end="\r")

    def close(self):
        pynvml.nvmlShutdown()


def find_run_dir(run_name: str) -> Path:
    """Finds the Ultralytics run folder whether saved under runs/detect/thesis_runs or thesis_runs."""
    candidates = [
        Path("runs") / "detect" / "thesis_runs" / run_name,
        Path("thesis_runs") / run_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def get_saved_train_time_sec(run_dir: Path, fallback_hours: float) -> float:
    """Reads total training time (seconds) from Ultralytics results.csv if training already finished."""
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        try:
            with open(results_csv, mode="r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    last_row = {k.strip(): v.strip() for k, v in rows[-1].items()}
                    if "time" in last_row:
                        return float(last_row["time"])
        except Exception:
            pass
    return fallback_hours * 3600.0


def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def append_row_to_csv(csv_path: Path, row: dict):
    write_header = not csv_path.exists()
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_already_logged_models(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    logged = set()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "Model" in row:
                logged.add(row["Model"].strip())
    return logged


def main():
    EXPORT_DIR.mkdir(exist_ok=True)
    already_logged = get_already_logged_models(OUTPUT_CSV)

    gpu_monitor = LaptopGPUMonitor(gpu_index=DEVICE, interval_sec=0.1)
    print(f"Detected GPU: {gpu_monitor.gpu_name} | Initial Temp: {gpu_monitor.get_current_temp()}°C")

    for idx, model_name in enumerate(MODELS):
        if model_name in already_logged:
            print(f"\n[{idx+1}/{len(MODELS)}] {model_name} is already logged in {OUTPUT_CSV}. Skipping!")
            continue

        is_yolo26 = "yolo26" in model_name
        batch_size = BATCH_MAP.get(model_name, 4)
        run_name = f"{model_name}_pea_1280"
        run_dir = find_run_dir(run_name)
        best_pt_path = run_dir / "weights" / "best.pt"
        last_pt_path = run_dir / "weights" / "last.pt"

        print(f"\n{'='*75}")
        print(f"[{idx+1}/{len(MODELS)}] Processing {model_name} | batch={batch_size} | NMS-Free={is_yolo26}")
        print(f"{'='*75}")

        # ------------------------------------------------------------------
        # 1. CHECK IF TRAINING ALREADY COMPLETED (OR NEEDS RESUME / START)
        # ------------------------------------------------------------------
        if best_pt_path.exists():
            print(f"Found existing completed weights at: {best_pt_path}")
            print(f"Skipping training for {model_name} and proceeding straight to validation & export!")
            train_time_s = get_saved_train_time_sec(run_dir, FALLBACK_YOLO11N_HOURS)
            train_avg_w = FALLBACK_TRAIN_WATTS
            train_peak_temp = float(gpu_monitor.get_current_temp())
            train_energy_wh = train_avg_w * (train_time_s / 3600.0)
        else:
            if idx > 0:
                gpu_monitor.wait_for_cooldown(target_temp_c=COOLDOWN_TARGET_TEMP_C, max_wait_sec=45)

            if last_pt_path.exists():
                print(f"Resuming partially finished training from: {last_pt_path}")
                model = YOLO(str(last_pt_path))
                gpu_monitor.start()
                t0_train = time.perf_counter()
                model.train(resume=True)
            else:
                model = YOLO(f"{model_name}.pt")
                gpu_monitor.start()
                t0_train = time.perf_counter()
                model.train(
                    data=DATA_YAML,
                    epochs=EPOCHS,
                    patience=PATIENCE,
                    imgsz=TRAIN_IMGSZ,
                    batch=batch_size,
                    rect=True,
                    cache=CACHE_MODE,
                    workers=DATALOADER_WORKERS,
                    amp=True,
                    close_mosaic=10,
                    device=DEVICE,
                    project=str(Path("runs/detect/thesis_runs").resolve()),
                    name=run_name,
                    exist_ok=True,
                )

            train_time_s = time.perf_counter() - t0_train
            train_avg_w, train_peak_w, train_peak_temp = gpu_monitor.stop()
            train_energy_wh = train_avg_w * (train_time_s / 3600.0)

            # Get exact path where Ultralytics saved best.pt
            best_pt_path = Path(model.trainer.best)
            del model
            cleanup_vram()

        trained_model = YOLO(str(best_pt_path))

        # ------------------------------------------------------------------
        # 2. VALIDATE ACCURACY METRICS ON LAPTOP (PyTorch FP16/AMP)
        # ------------------------------------------------------------------
        try:
            val_metrics = trained_model.val(
                data=DATA_YAML,
                imgsz=TRAIN_IMGSZ,
                batch=batch_size,
                rect=True,
                split=EVAL_SPLIT,
                workers=4,
                device=DEVICE,
            )
        except Exception:
            val_metrics = trained_model.val(
                data=DATA_YAML,
                imgsz=TRAIN_IMGSZ,
                batch=batch_size,
                rect=True,
                split="val",
                workers=4,
                device=DEVICE,
            )

        map50 = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
        precision = float(val_metrics.box.mp)
        recall = float(val_metrics.box.mr)

        # Brief thermal stabilization before 300-frame latency & power test
        gpu_monitor.wait_for_cooldown(target_temp_c=COOLDOWN_TARGET_TEMP_C, max_wait_sec=30)

        # ------------------------------------------------------------------
        # 3. BENCHMARK INFERENCE SPEED, POWER & JOULES/FRAME (960x1280)
        # ------------------------------------------------------------------
        for _ in range(WARMUP_FRAMES):
            trained_model.predict(
                SAMPLE_IMAGE,
                imgsz=INFER_IMGSZ,
                half=True,
                device=DEVICE,
                nms=not is_yolo26,
                verbose=False,
            )
        torch.cuda.synchronize()

        gpu_monitor.start()
        t0_infer = time.perf_counter()
        for _ in range(BENCHMARK_FRAMES):
            trained_model.predict(
                SAMPLE_IMAGE,
                imgsz=INFER_IMGSZ,
                half=True,
                device=DEVICE,
                nms=not is_yolo26,
                verbose=False,
            )
        torch.cuda.synchronize()
        infer_time_s = time.perf_counter() - t0_infer
        infer_avg_w, infer_peak_w, infer_peak_temp = gpu_monitor.stop()

        fps = BENCHMARK_FRAMES / infer_time_s
        latency_ms = (infer_time_s / BENCHMARK_FRAMES) * 1000.0
        joules_per_frame = infer_avg_w / fps

        # ------------------------------------------------------------------
        # 4. EXPORT PORTABLE WEIGHTS (.pt and .onnx) FOR JETSON ORIN NANO
        # ------------------------------------------------------------------
        shutil.copy2(best_pt_path, EXPORT_DIR / f"{model_name}_best.pt")
        onnx_path = trained_model.export(
            format="onnx",
            imgsz=INFER_IMGSZ,
            simplify=True,
            opset=12,
        )
        shutil.move(str(onnx_path), EXPORT_DIR / f"{model_name}_best.onnx")

        row = {
            "Platform": f"Laptop ({gpu_monitor.gpu_name})",
            "Model": model_name,
            "Precision": "PyTorch FP16",
            "Resolution": f"{INFER_IMGSZ[1]}x{INFER_IMGSZ[0]}",
            "NMS_Free": is_yolo26,
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Latency_ms": round(latency_ms, 2),
            "FPS": round(fps, 2),
            "Avg_Infer_Power_W": round(infer_avg_w, 2),
            "Peak_Infer_Power_W": round(infer_peak_w, 2),
            "Energy_Joules_per_Frame": round(joules_per_frame, 4),
            "Train_Time_min": round(train_time_s / 60.0, 2),
            "Train_Avg_Power_W": round(train_avg_w, 2),
            "Train_Energy_Wh": round(train_energy_wh, 2),
            "Peak_GPU_Temp_C": round(max(train_peak_temp, infer_peak_temp), 1),
        }
        append_row_to_csv(OUTPUT_CSV, row)
        print(f"Saved result for {model_name} -> {row}")

        del trained_model
        cleanup_vram()

    gpu_monitor.close()
    print(f"\nAll 6 models complete! Saved full results to: {OUTPUT_CSV}")
    print(f"Exported weights for Jetson Orin Nano are ready in: {EXPORT_DIR}/")


if __name__ == "__main__":
    main()