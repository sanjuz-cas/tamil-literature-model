print("--- [Sanity Check] Starting ---")
try:
    import torch, transformers, wandb
    import train

    assert hasattr(train, "main"), "train.py missing main()"
    print("✅ All checks passed. Ready to build Docker image.")
except Exception as e:
    print("❌ Sanity check failed:", e)
    exit(1)
