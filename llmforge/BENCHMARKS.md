# Benchmarks

## Commands 

The command for evaluating on MMLU Pro:

```
llmforge.evaluate --model model/repo --task mmlu_pro
```
 
The command for efficiency benchmarks:

```
llmforge.benchmark --model model/repo -p 2048 -g 128
```

To get the package versions run:

```
python -m llmforge --version
```

## Models

<details>

 <summary> Qwen/Qwen3-4B-Instruct-2507 </summary>

Precision | MMLU Pro | Prompt (2048) tok/sec | Generation (128) tok/sec | Memory GB | Repo
--------- | -------- | ------------------- | ------------------------ | --------- | ----
bf16      | 64.05    | 1780.63             | 52.47                    | 9.02    | Qwen/Qwen3-4B-Instruct-2507
q8 | 63.85 | 1606.573| 86.907 | 5.254 | hf-user/Qwen3-4B-Instruct-2507-8bit
q6 | 63.53 | 1576.73 | 104.68 | 4.25 | hf-user/Qwen3-4B-Instruct-2507-6bit
q5 g32 | 63.16 | 1570.80 | 110.29 | 4.00 | hf-user/Qwen3-4B-Instruct-2507-5bit-g32
q5 | 62.38 | 1584.33 | 116.39 | 3.86 | hf-user/Qwen3-4B-Instruct-2507-5bit
q4 g32 | 61.46 | 1610.03 | 126.00 | 3.603 | hf-user/Qwen3-4B-Instruct-2507-4bit-g32
q4 | 60.72 | 1622.27 | 134.52 | 3.35 | hf-user/Qwen3-4B-Instruct-2507-4bit

- Performance benchmark on 64GB GPU
- llmforge 0.1.0
- PyTorch 2.0+
 
</details>

<details>
<summary> Qwen/Qwen3-30B-A3B-Instruct-2507 </summary>

Precision | MMLU Pro | Prompt (2048) tok/sec | Generation (128) tok/sec | Memory GB | Repo
--------- | -------- | ------------------- | ------------------------ | --------- | ----
bf16 | 72.62 | :skull: | :skull: | :skull: | Qwen/Qwen3-30B-A3B-Instruct-2507
q8 | 72.46 | 1719.47 | 83.16 | 33.46 | hf-user/Qwen3-30B-A3B-Instruct-2507-8bit 
q6 | 72.41 | 1667.45 | 94.14 | 25.82 | hf-user/Qwen3-30B-A3B-Instruct-2507-6bit
q5 | 71.97 | 1664.24 | 101.00 |22.01 | hf-user/Qwen3-30B-A3B-Instruct-2507-5bit
q4 | 70.71 | 1753.90 | 113.33 |18.20 | hf-user/Qwen3-30B-A3B-Instruct-2507-4bit

 
- Performance benchmarks on 64GB GPU
- llmforge 0.1.0
- PyTorch 2.0+

</details>
