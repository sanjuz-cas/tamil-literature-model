import os

print("--- [Fast Test] Starting Sanity Check ---")

try:
    # 1. Test: Can we import the training script?
    # This will catch any simple syntax errors or bad imports
    # *that are not cloud-specific*.
    import train

    print("[SUCCESS] train.py imported successfully.")

    # 2. Test: Does the script have the main function?
    assert hasattr(train, "main"), "train.py is missing the 'main' function!"
    print("[SUCCESS] 'main' function found.")

    # 3. Test: Can we import the main local libraries?
    import transformers
    import datasets

    print("[SUCCESS] All major local libraries are importable.")

    print("\n--- [Fast Test] All local checks passed! ---")
    print("Your code is ready to be built into a Docker container.")

except Exception as e:
    print(f"\n--- [FAILED] Test failed with error: ---")
    print(e)
    # Exit with an error code to stop any automation
    exit(1)
