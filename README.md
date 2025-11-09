# Tamil LLM (2B parameters)

This repo trains a 2B-parameter Tamil language model based on `google/gemma-2b`.

## Commands

```bash
# Build locally
docker build -t us-central1-docker.pkg.dev/tpu-research-sanju/tamil-llm/gemma-2b:latest .

# Push manually
docker push us-central1-docker.pkg.dev/tpu-research-sanju/tamil-llm/gemma-2b:latest
