list of models to try for zero shot and few shot experimenets

- Qwen 3 family
- Kimi K2 family
- GPT OSS
- Llama 3.1
- Mistral 3


vllm setup instructions
- install uv
- uv venv
- uv pip install vllm --torch-backend=auto
- 


```
cd /tmp
pip install uv
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
pwd
vllm serve Qwen/Qwen3-8B
```

```
pip install huggingface-hub==0.36.0
hf auth login
```
