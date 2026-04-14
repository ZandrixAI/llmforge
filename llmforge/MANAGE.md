# Managing Models

You can use `llmforge` to manage models downloaded locally in your machine. They
are stored in the Hugging Face cache.

Scan models: 

```shell
llmforge.manage --scan
```

Specify a `--pattern` to get info on a single or specific set of models:

```shell
llmforge.manage --scan --pattern hf-user/Mistral-7B-Instruct-v0.2-4bit
```

To delete a model (or multiple models):

```shell
llmforge.manage --delete --pattern hf-user/Mistral-7B-Instruct-v0.2-4bit
```
