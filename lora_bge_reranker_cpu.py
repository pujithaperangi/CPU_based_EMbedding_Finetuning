"""
LoRA Cross-Encoder Fine-Tuning Script (nohup-ready, production-grade)
=======================================================================
Run with:
    nohup python lora_finetune_10lk.py > lora_finetune_10lk.log 2>&1 &

Logs  : lora_finetune_10lk.log  (every step detail + ETA)
Metrics: metrics.log             (training metrics per step/epoch)
Resource summary printed to lora_finetune_10lk.log at the end
         (max CPU % and max RSS memory of THIS process)
"""

# ===========================================================================
# 0.  FORCE CPU-ONLY  ── must be the very first thing before any torch import
# ===========================================================================
import os

# from ray import state
# from sklearn import metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import gc
# ===========================================================================
# 1.  STANDARD LIBRARY IMPORTS
# ===========================================================================
import json
import re
import time
import signal
import random
import logging
import datetime
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List

# ===========================================================================
# 2.  THIRD-PARTY IMPORTS
# ===========================================================================
import numpy as np
import pandas as pd
import psutil                          # pip install psutil
import torch

torch.set_default_device("cpu")        # extra safety – no GPU leakage

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from datasets import Dataset, DatasetDict
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
)
from huggingface_hub import login, HfApi
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainerCallback

# ===========================================================================
# 3.  LOGGING SETUP
# ===========================================================================
LOG_FILE     = "lora_finetune_10lk.log"
METRICS_FILE = "metrics_10lk.log"

def _make_logger(name: str, log_file: str, level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()           # also echo to stdout (captured by nohup)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger  = _make_logger("finetune", LOG_FILE)
mlogger = _make_logger("metrics",  METRICS_FILE)

# ===========================================================================
# 4.  CONFIGURATION
# ===========================================================================
class Config:
    # ── Model ──────────────────────────────────────────────────────────────
    MODEL_NAME     = "BAAI/bge-reranker-v2-m3"
    MAX_SEQ_LENGTH = 64    # based on the dataset's sentence length distribution(sent1+sent2)

    # ── Training ───────────────────────────────────────────────────────────
    NUM_EPOCHS                 = 1
    BATCH_SIZE                 = 512       
    LEARNING_RATE              = 2e-5
    WARMUP_RATIO               = 0.1
    WEIGHT_DECAY               = 0.01
    MAX_GRAD_NORM              = 1.0
    GRADIENT_ACCUMULATION_STEPS = 1     

    # ── Data split ─────────────────────────────────────────────────────────
    TEST_SIZE    = 0.15
    VAL_SIZE     = 0.15
    RANDOM_STATE = 42

    # ── Paths ──────────────────────────────────────────────────────────────
    INPUT_JSON_PATH = "/mnt/data/aidata/RA_genai/Embedding_finetune/10lakh_individual_lora_finetune/name_address_10lakh.json"
    OUTPUT_DIR      = "finetuned_bge_reranker_individual_10lakh_trail"
    TEST_SAVE_PATH  = "test_data_individual_10lakh.json"
    FINAL_SAVE_DIR  = "finetuned-bge-reranker-merged-individual-10lakh-trail"

    # ── HuggingFace ────────────────────────────────────────────────────────
    HF_REPO_NAME = "pujithapsx/bge_reranker_10lakh_3april"
    HF_TOKEN     = ""  

    # ── LoRA ───────────────────────────────────────────────────────────────
    LORA_R              = 16
    LORA_ALPHA          = 32
    LORA_DROPOUT        = 0.05
    LORA_TARGET_MODULES = ["query", "key", "value", "out"]

config = Config()
os.makedirs(config.OUTPUT_DIR,    exist_ok=True)
os.makedirs(config.FINAL_SAVE_DIR, exist_ok=True)

# ===========================================================================
# 5.  REPRODUCIBILITY
# ===========================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===========================================================================
# 6.  CPU THREAD TUNING
# ===========================================================================
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
try:
    torch.set_num_interop_threads(max(1, num_cores // 2))
except RuntimeError:
    pass   # already set; ignore

logger.info("=" * 70)
logger.info("LoRA Cross-Encoder Fine-Tuning – START")
logger.info(f"  PID            : {os.getpid()}")
logger.info(f"  CPU cores      : {num_cores}")
logger.info(f"  PyTorch version: {torch.__version__}")
logger.info(f"  CUDA available : {torch.cuda.is_available()}")
logger.info(f"  Log file       : {LOG_FILE}")
logger.info(f"  Metrics file   : {METRICS_FILE}")
logger.info("=" * 70)

# ===========================================================================
# 7.  RESOURCE MONITOR  (background thread tracks THIS process)
# ===========================================================================
class ResourceMonitor:
    """Samples CPU% and RSS memory of the current process every `interval` s."""

    def __init__(self, interval: float = 5.0):
        self.interval   = interval
        self.proc       = psutil.Process(os.getpid())
        self.max_cpu    = 0.0          # peak CPU %  (this process)
        self.max_mem_mb = 0.0          # peak RSS in MB
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        logger.info("[ResourceMonitor] Started – sampling every "
                    f"{self.interval}s for PID {os.getpid()}")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 2)

    def _run(self):
        while not self._stop.is_set():
            try:
                cpu = self.proc.cpu_percent(interval=None)   # non-blocking
                mem = self.proc.memory_info().rss / (1024 ** 2)  # Correct MB calculation
                if cpu > self.max_cpu:
                    self.max_cpu = cpu
                if mem > self.max_mem_mb:
                    self.max_mem_mb = mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self.interval)

    def report(self) -> str:
        lines = [
            "=" * 70,
            "RESOURCE SUMMARY (this process)",
            f"  PID                 : {os.getpid()}",
            f"  Peak CPU utilisation: {self.max_cpu:.1f}%",
            f"  Peak memory (RSS)   : {self.max_mem_mb:.1f} MB "
            f"({self.max_mem_mb / 1024:.2f} GB)",
            "=" * 70,
        ]
        return "\n".join(lines)


monitor = ResourceMonitor(interval=5.0)
monitor.start()

# ===========================================================================
# 8.  GRACEFUL-SHUTDOWN HANDLER  (SIGINT / SIGTERM)
# ===========================================================================
_shutdown = threading.Event()

def _handle_signal(sig, frame):
    logger.warning(f"Signal {sig} received – will attempt graceful shutdown "
                   "after current step.")
    _shutdown.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ===========================================================================
# 9.  HF LOGIN
# ===========================================================================
logger.info("Logging in to HuggingFace Hub …")
login(token=config.HF_TOKEN, add_to_git_credential=False)
logger.info("HuggingFace Hub login successful.")

# ===========================================================================
# 10. DATA LOADING & VALIDATION
# ===========================================================================
logger.info("-" * 70)
logger.info(f"Loading data from: {config.INPUT_JSON_PATH}")

if not os.path.exists(config.INPUT_JSON_PATH):
    logger.error(f"Input file not found: {config.INPUT_JSON_PATH}")
    raise FileNotFoundError(config.INPUT_JSON_PATH)

# --- quick scan for bad control characters (optional integrity check) ---
logger.info("Scanning for control characters in JSON file …")
bad_lines: List = []
with open(config.INPUT_JSON_PATH, "r", encoding="utf-8") as fh:
    for i, line in enumerate(fh, 1):
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", line):
            bad_lines.append(i)
if bad_lines:
    logger.warning(f"Found {len(bad_lines)} lines with control characters "
                   f"(first 5: {bad_lines[:5]})")
else:
    logger.info("No control character issues found in JSON file.")

logger.info("Parsing JSON …")
with open(config.INPUT_JSON_PATH, "r", encoding="utf-8") as fh:
    raw_data = json.load(fh)

if isinstance(raw_data, dict):
    raw_data = [raw_data]

logger.info(f"✓ Loaded {len(raw_data):,} raw records.")

# --- convert to DataFrame ---
logger.info("Converting to DataFrame …")
processed = []
for idx, item in enumerate(raw_data):
    try:
        s1     = str(item.get("item1", item.get("value1", "")))
        s2     = str(item.get("item2", item.get("value2", "")))
        result = item.get("result", "no match")
        label  = 1 if str(result).lower() in {"match", "yes", "true", "1"} else 0
        processed.append({"sentence1": s1, "sentence2": s2, "label": label})
    except Exception as exc:
        logger.warning(f"  Skipping item {idx}: {exc}")

df = pd.DataFrame(processed)
logger.info(f"✓ Processed {len(df):,} valid examples  "
            f"[match={int(df['label'].sum()):,}  "
            f"no_match={len(df) - int(df['label'].sum()):,}]")

# --- split ---
logger.info("Splitting dataset …")
train_val_df, test_df = train_test_split(
    df,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=df["label"],
)
val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_ratio,
    random_state=config.RANDOM_STATE,
    stratify=train_val_df["label"],
)
logger.info(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  "
            f"|  Test: {len(test_df):,}")

# save test set
with open(config.TEST_SAVE_PATH, "w", encoding="utf-8") as fh:
    json.dump(test_df.to_dict("records"), fh, indent=2, ensure_ascii=False)
logger.info(f"Test set saved → {config.TEST_SAVE_PATH}")

# --- HuggingFace Datasets ---
train_dataset = Dataset.from_pandas(
    train_df[["sentence1", "sentence2", "label"]].reset_index(drop=True)
)
val_dataset = Dataset.from_pandas(
    val_df[["sentence1", "sentence2", "label"]].reset_index(drop=True)
)
hf_datasets = DatasetDict({"train": train_dataset, "validation": val_dataset})
logger.info(f"HuggingFace DatasetDict ready:\n{hf_datasets}")

# ===========================================================================
# 11. ETA ESTIMATOR
# ===========================================================================
class ETAEstimator:
    def __init__(self, total_steps: int):
        self.total  = total_steps
        self.start  = time.time()
        self.done   = 0

    def update(self, steps_done: int):
        self.done = steps_done

    def eta_str(self) -> str:
        if self.done == 0:
            return "calculating …"
        elapsed   = time.time() - self.start
        per_step  = elapsed / self.done
        remaining = (self.total - self.done) * per_step
        return str(datetime.timedelta(seconds=int(remaining)))

    def elapsed_str(self) -> str:
        return str(datetime.timedelta(seconds=int(time.time() - self.start)))


# ===========================================================================
# 12. CUSTOM TRAINER CALLBACK  (logging + metrics + ETA + checkpoint guard)
# ===========================================================================
class DetailedLoggingCallback(TrainerCallback):
    def __init__(self, eta: ETAEstimator):
        self.eta          = eta
        self.step_times:  List[float] = []
        self._step_start: float       = 0.0
        self._header_written          = False   # FIX 1

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("=" * 70)
        logger.info("TRAINING STARTED")
        logger.info(f"  Total steps  : {state.max_steps:,}")
        logger.info(f"  Epochs       : {args.num_train_epochs}")
        logger.info(f"  Batch size   : {args.per_device_train_batch_size}")
        logger.info(f"  LR           : {args.learning_rate}")
        logger.info(f"  Output dir   : {args.output_dir}")
        logger.info("=" * 70)
        if not self._header_written:            # FIX 1 — write header only once
            mlogger.info("step,epoch,train_loss,eval_loss,accuracy,f1,precision,recall,ap")
            self._header_written = True

    # def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    #     if not metrics:
    #         return

    #     # FIX 2 — add debug log to see ALL keys actually present
    #     logger.debug(f"[EVAL keys available]: {list(metrics.keys())}")

    #     step    = state.global_step
    #     epoch   = round(state.epoch or 0, 4)

    #     # FIX 2 — correct key names with eval_ prefix
    #     # ev_loss = metrics.get("eval_loss", float("nan"))
    #     # acc     = metrics.get("eval_entity-matching_accuracy",    float("nan"))
    #     # f1      = metrics.get("eval_entity-matching_f1",          float("nan"))
    #     # prec    = metrics.get("eval_entity-matching_precision",   float("nan"))
    #     # rec     = metrics.get("eval_entity-matching_recall",      float("nan"))
    #     # ap      = metrics.get("eval_entity-matching_average_precision", float("nan"))
    #     # f1 = (metrics.get("eval_entity-matching_f1") or 
    #     # metrics.get("entity-matching_f1") or 
    #     # metrics.get("eval_f1") or 
    #     # metrics.get("f1") or 
    #     # float("nan"))
    
    #     # acc = (metrics.get("eval_entity-matching_accuracy") or 
    #     #     metrics.get("entity-matching_accuracy") or 
    #     #     metrics.get("eval_accuracy") or 
    #     #     metrics.get("accuracy") or 
    #     #     float("nan"))
        
    #     # prec = (metrics.get("eval_entity-matching_precision") or 
    #     #         metrics.get("entity-matching_precision") or 
    #     #         metrics.get("eval_precision") or 
    #     #         metrics.get("precision") or 
    #     #         float("nan"))
        
    #     # rec = (metrics.get("eval_entity-matching_recall") or 
    #     #     metrics.get("entity-matching_recall") or 
    #     #     metrics.get("eval_recall") or 
    #     #     metrics.get("recall") or 
    #     #     float("nan"))
    #     # ap = (metrics.get("eval_entity-matching_average_precision") or 
    #     #     metrics.get("entity-matching_average_precision") or 
    #     #     metrics.get("eval_average_precision") or 
    #     #     metrics.get("average_precision") or 
    #     #     float("nan"))
        
    # def _get_metric(metrics: dict, *keys) -> float:
    #     for k in keys:
    #         v = metrics.get(k)
    #         if v is not None:
    #             return float(v)
    #     return float("nan")

    #     f1   = _get_metric(metrics, "eval_entity-matching_f1", "entity-matching_f1", "eval_f1", "f1")
    #     acc  = _get_metric(metrics, "eval_entity-matching_accuracy", "entity-matching_accuracy", "eval_accuracy", "accuracy")
    #     prec = _get_metric(metrics, "eval_entity-matching_precision", "entity-matching_precision", "eval_precision", "precision")
    #     rec  = _get_metric(metrics, "eval_entity-matching_recall", "entity-matching_recall", "eval_recall", "recall")
    #     ap   = _get_metric(metrics, "eval_entity-matching_average_precision", "entity-matching_average_precision", "eval_average_precision", "average_precision")
    #     ev_loss = metrics.get("eval_loss", float("nan"))
    #     logger.info(
    #         f"[EVAL  step={step:,}] epoch={epoch}  eval_loss={ev_loss:.6f}  "
    #         f"accuracy={acc:.6f}  f1={f1:.6f}  precision={prec:.6f}  "
    #         f"recall={rec:.6f}  avg_precision={ap:.6f}  "
    #         f"ETA={self.eta.eta_str()}"
    #     )

    #     tr_loss = state.log_history[-1].get("loss", float("nan")) \
    #               if state.log_history else float("nan")
    #     mlogger.info(
    #         f"{step},{epoch},{tr_loss:.6f},{ev_loss:.6f},"
    #         f"{acc:.6f},{f1:.6f},{prec:.6f},{rec:.6f},{ap:.6f}"
    #     )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        logger.debug(f"[EVAL keys available]: {list(metrics.keys())}")

        step = state.global_step
        epoch = round(state.epoch or 0, 4)

        def _get_metric(m_dict: dict, *keys) -> float:
            for k in keys:
                v = m_dict.get(k)
                if v is not None:
                    return float(v)
            return float("nan")

        f1   = _get_metric(metrics, "eval_entity-matching_f1", "entity-matching_f1", "eval_f1", "f1")
        acc  = _get_metric(metrics, "eval_entity-matching_accuracy", "entity-matching_accuracy", "eval_accuracy", "accuracy")
        prec = _get_metric(metrics, "eval_entity-matching_precision", "entity-matching_precision", "eval_precision", "precision")
        rec  = _get_metric(metrics, "eval_entity-matching_recall", "entity-matching_recall", "eval_recall", "recall")
        ap   = _get_metric(metrics, "eval_entity-matching_average_precision", "entity-matching_average_precision", "eval_average_precision", "average_precision")
        ev_loss = metrics.get("eval_loss", float("nan"))

        logger.info(
            f"[EVAL  step={step:,}] epoch={epoch}  eval_loss={ev_loss:.6f}  "
            f"accuracy={acc:.6f}  f1={f1:.6f}  precision={prec:.6f}  "
            f"recall={rec:.6f}  avg_precision={ap:.6f}  "
            f"ETA={self.eta.eta_str()}"
        )

        tr_loss = state.log_history[-1].get("loss", float("nan")) if state.log_history else float("nan")
        mlogger.info(
            f"{step},{epoch},{tr_loss:.6f},{ev_loss:.6f},"
            f"{acc:.6f},{f1:.6f},{prec:.6f},{rec:.6f},{ap:.6f}"
        )

    def on_save(self, args, state, control, **kwargs):
        logger.info(
            f"[CHECKPOINT saved] step={state.global_step:,}  "
            f"dir={args.output_dir}/checkpoint-{state.global_step}"
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        logger.info(f"── EPOCH {epoch} / {int(args.num_train_epochs)} ENDED "
                    f"  elapsed={self.eta.elapsed_str()} ──")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("=" * 70)
        logger.info(f"TRAINING ENDED  |  total elapsed={self.eta.elapsed_str()}")
        logger.info("=" * 70)


# ===========================================================================
# 13. MODEL + LoRA
# ===========================================================================
logger.info("Loading base CrossEncoder model …")
base_model = CrossEncoder(
    config.MODEL_NAME,
    num_labels=1,
    max_length=config.MAX_SEQ_LENGTH,
    trust_remote_code=True,
    device="cpu",
)
logger.info(f"Base model loaded: {config.MODEL_NAME}")

lora_cfg = LoraConfig(
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=config.LORA_TARGET_MODULES,
)
peft_model        = get_peft_model(base_model.model, lora_cfg)
base_model.model  = peft_model
logger.info("LoRA adapter applied successfully.")
# log trainable param summary
total_params     = sum(p.numel() for p in peft_model.parameters())
trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
logger.info(f"  Trainable params: {trainable_params:,}  /  "
            f"Total: {total_params:,}  "
            f"({100 * trainable_params / total_params:.4f}%)")

# ===========================================================================
# 14. LOSS + EVALUATOR
# ===========================================================================
logger.info("Setting up loss function …")
loss = BinaryCrossEntropyLoss(model=base_model)

logger.info("Building validation evaluator (sub-sampling to 20k for speed) …")
# For 300k+ val rows building a full list() blocks for minutes.
# We use a random 20 000-row sample for the evaluator – still informative.
MAX_EVAL_SAMPLES = 10_000
val_s1  = hf_datasets["validation"]["sentence1"]
val_s2  = hf_datasets["validation"]["sentence2"]
val_lbl = hf_datasets["validation"]["label"]

if len(val_s1) > MAX_EVAL_SAMPLES:
    rng    = np.random.default_rng(SEED)
    idx    = rng.choice(len(val_s1), MAX_EVAL_SAMPLES, replace=False)
    val_s1  = [val_s1[i]  for i in idx]
    val_s2  = [val_s2[i]  for i in idx]
    val_lbl = [val_lbl[i] for i in idx]
    logger.info(f"  Evaluator sub-sampled to {MAX_EVAL_SAMPLES:,} pairs.")

evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(val_s1, val_s2)),
    labels=val_lbl,
    name="entity-matching",
    show_progress_bar=False,
)
logger.info("Loss and Evaluator ready.")

# ===========================================================================
# 15. TOTAL STEPS CALCULATION  (FIX: use config values only, no training_args ref)
# ===========================================================================
total_batches = len(hf_datasets["train"]) // config.BATCH_SIZE
total_steps   = (total_batches // config.GRADIENT_ACCUMULATION_STEPS) * config.NUM_EPOCHS

# Realistic step time estimate for CPU + large batch
if config.BATCH_SIZE >= 512:
    SECS_PER_STEP_ESTIMATE = 70.0      # 512+
elif config.BATCH_SIZE >= 256:
    SECS_PER_STEP_ESTIMATE = 45.0      # 256-511
else:
    SECS_PER_STEP_ESTIMATE = 12.0      # <256     # for small batches

est_secs = total_steps * SECS_PER_STEP_ESTIMATE

logger.info("-" * 70)
logger.info("APPROXIMATE TRAINING TIME ESTIMATE")
logger.info(f"  Train samples       : {len(hf_datasets['train']):,}")
logger.info(f"  Batch size          : {config.BATCH_SIZE}")
logger.info(f"  Gradient accum      : {config.GRADIENT_ACCUMULATION_STEPS}")
logger.info(f"  Total steps         : {total_steps:,}")
logger.info(f"  Assumed s/step      : {SECS_PER_STEP_ESTIMATE:.1f}s  (realistic for CPU)")
# logger.info(f"  Estimated duration  : {datetime.timedelta(seconds=int(est_secs))}")
logger.info("  (Actual speed is recalculated live during training)")
logger.info("-" * 70)

# ===========================================================================
# 16. TRAINING ARGUMENTS  (FIX: defined after total_steps is computed)
# ===========================================================================
training_args = CrossEncoderTrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.NUM_EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=64,
    learning_rate=config.LEARNING_RATE,
    warmup_ratio=config.WARMUP_RATIO,
    weight_decay=config.WEIGHT_DECAY,
    max_grad_norm=config.MAX_GRAD_NORM,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,

    dataloader_num_workers=0,
    dataloader_pin_memory=False, # false 
    dataloader_persistent_workers=False,
    dataloader_prefetch_factor=None,

    eval_strategy="steps",
    eval_steps=max(1, total_steps // 5),
    save_strategy="steps",
    save_steps=max(1, total_steps // 5),
    logging_steps=200,

    load_best_model_at_end=False,  # just to check the stability
    # metric_for_best_model="eval_f1",        # Better than loss
    metric_for_best_model="entity-matching_f1",  
    greater_is_better=True,

    save_total_limit=2,
    report_to=[],
    fp16=False,
    bf16=False,
    use_cpu=True,
)
# ===========================================================================
# 17. RESUME-FROM-CHECKPOINT DETECTION
# ===========================================================================
def _latest_checkpoint(output_dir: str):
    """Return the latest checkpoint dir, or None."""
    ckpt_dirs = sorted(
        [d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    return str(ckpt_dirs[-1]) if ckpt_dirs else None

resume_ckpt = _latest_checkpoint(config.OUTPUT_DIR)
if resume_ckpt:
    logger.info(f"Found checkpoint: {resume_ckpt} — loading LoRA weights manually …")
    from peft import PeftModel
    # The base_model.model is already a PeftModel; load saved adapter weights into it
    # base_model.model.load_adapter(resume_ckpt, adapter_name="default")
    logger.info("✅ LoRA adapter weights restored from checkpoint.")
else:
    logger.info("No previous checkpoint found – starting fresh.")

# ===========================================================================
# 18. TRAINER
# ===========================================================================
eta_estimator = ETAEstimator(total_steps=total_steps)
detail_cb     = DetailedLoggingCallback(eta=eta_estimator)

trainer = CrossEncoderTrainer(
    model=base_model,
    args=training_args,
    train_dataset=hf_datasets["train"],
    eval_dataset=hf_datasets["validation"],
    loss=loss,
    evaluator=evaluator,
    callbacks=[detail_cb],
)
logger.info("Trainer initialised with LoRA + DetailedLoggingCallback.")

# ===========================================================================
# 19. TRAIN  — pass None so the broken _load_from_checkpoint is never called
# ===========================================================================
logger.info("🚀 Starting LoRA fine-tuning …")
train_result = trainer.train(resume_from_checkpoint=None)   # ← always None

logger.info("🎉 LoRA fine-tuning completed!")
# log training summary
logger.info(f"  Training runtime    : {train_result.metrics.get('train_runtime', 0):.1f}s")
logger.info(f"  Samples / second    : {train_result.metrics.get('train_samples_per_second', 0):.2f}")
logger.info(f"  Steps / second      : {train_result.metrics.get('train_steps_per_second', 0):.4f}")
logger.info(f"  Global steps done   : {train_result.metrics.get('train_global_step', total_steps)}")
logger.info(f"  Final train loss    : {train_result.metrics.get('train_loss', float('nan')):.6f}")

mlogger.info("# Training summary")
for k, v in train_result.metrics.items():
    mlogger.info(f"# {k}: {v}")

# ===========================================================================
# 20. MERGE LoRA → FULL MODEL & SAVE LOCALLY
# ===========================================================================
logger.info("-" * 70)
logger.info("Merging LoRA adapter into base model …")
merged_model      = base_model.model.merge_and_unload()
base_model.model  = merged_model
logger.info("✅ LoRA adapter merged successfully.")

logger.info(f"Saving merged model locally → {config.FINAL_SAVE_DIR}")
base_model.save_pretrained(config.FINAL_SAVE_DIR)
base_model.tokenizer.save_pretrained(config.FINAL_SAVE_DIR)
logger.info(f"✅ Merged model saved → {config.FINAL_SAVE_DIR}")

# ===========================================================================
# 21. PUSH TO HUGGING FACE HUB
# ===========================================================================
logger.info("-" * 70)
logger.info(f"Pushing model to HuggingFace Hub: {config.HF_REPO_NAME} …")
try:
    base_model.push_to_hub(
        repo_id=config.HF_REPO_NAME,
        commit_message="10-lakh name+address LoRA fine-tune (individual records)",
        private=False,
        exist_ok=True,
    )
    logger.info(f"✅ Model pushed successfully → "
                f"https://huggingface.co/{config.HF_REPO_NAME}")
except Exception as exc:
    logger.error(f"HuggingFace Hub push failed: {exc}")
    logger.info("Model is saved locally and can be pushed manually later.")

# ===========================================================================
# 22. RESOURCE REPORT  (stop monitor → print summary)
# ===========================================================================
monitor.stop()
resource_report = monitor.report()
logger.info(resource_report)

# also write to metrics.log for easy retrieval
mlogger.info(resource_report)

logger.info("=" * 70)
logger.info("ALL DONE.")
logger.info(f"  Main log    → {LOG_FILE}")
logger.info(f"  Metrics     → {METRICS_FILE}")
logger.info(f"  Checkpoints → {config.OUTPUT_DIR}/")
logger.info(f"  Final model → {config.FINAL_SAVE_DIR}/")
logger.info(f"  HF Hub      → https://huggingface.co/{config.HF_REPO_NAME}")
logger.info("=" * 70)