"""Sequential pipeline after the TavBERT fine-tune that is already running:

1. wait for models/tavbert_full_ppp_nonbib_local export
2. fine-tune MsBERT (same headless driver)
3. run the external eval_runner on the FULL test split (limit=None) for:
   base TavBERT, fine-tuned TavBERT, base MsBERT, fine-tuned MsBERT

Results land in full_test_eval_results/.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent.parent / ".venv" / "bin" / "python"
TUNING = HERE.parent / "external_impl" / "new_dead_sea_scrolls" / "tuning"
DATA = (
    HERE.parent
    / "external_impl"
    / "new_dead_sea_scrolls"
    / "data_preparation"
    / "dss_sentences_min7_splits_ppp_nonbib.xlsx"
)
OUT = HERE / "full_test_eval_results"

TAVBERT_EXPORT = HERE / "models" / "tavbert_full_ppp_nonbib_local"
MSBERT_EXPORT = HERE / "models" / "msbert_full_ppp_nonbib_local"


def wait_for(path, label, timeout_s=4 * 3600):
    t0 = time.time()
    while not (path / "model.safetensors").exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"gave up waiting for {label} at {path}")
        time.sleep(30)
    time.sleep(15)  # let tokenizer files finish writing
    print(f"READY: {label}", flush=True)


def finetune(run_name, model_id):
    print(f"FINETUNE START: {run_name}", flush=True)
    subprocess.run(
        [str(PY), str(HERE / "run_finetune_headless.py"), run_name, model_id],
        cwd=HERE,
        check=True,
    )
    print(f"FINETUNE OK: {run_name}", flush=True)


def evaluate(model_id, tag):
    print(f"EVAL START: {tag} ({model_id})", flush=True)
    sys.path.insert(0, str(TUNING))
    from eval_runner import run_eval

    run_eval(
        model_id=str(model_id),
        dataset_path=str(DATA),
        out_root=str(OUT),
        split="test",
        limit=None,
        push_to_hub=False,
    )
    print(f"EVAL OK: {tag}", flush=True)


def main():
    wait_for(TAVBERT_EXPORT, "tavbert fine-tune export")
    if not (MSBERT_EXPORT / "model.safetensors").exists():
        finetune("msbert_full_ppp_nonbib_local", "dicta-il/MsBERT")

    OUT.mkdir(exist_ok=True)
    for model_id, tag in [
        ("tau/tavbert-he", "tavbert-base"),
        (TAVBERT_EXPORT, "tavbert-finetuned"),
        ("dicta-il/MsBERT", "msbert-base"),
        (MSBERT_EXPORT, "msbert-finetuned"),
    ]:
        evaluate(model_id, tag)

    print("PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
