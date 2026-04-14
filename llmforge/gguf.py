import re
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from transformers import AutoTokenizer


class TokenType(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class GGMLFileType(IntEnum):
    GGML_TYPE_F16 = 1


class HfVocab:
    def __init__(
        self,
        fname_tokenizer: Path,
        fname_added_tokens: Optional[Union[Path, None]] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            fname_tokenizer,
            cache_dir=fname_tokenizer,
            local_files_only=True,
        )
        self.added_tokens_list = []
        self.added_tokens_dict = dict()
        self.added_tokens_ids = set()
        for tok, tokidx in sorted(
            self.tokenizer.get_added_vocab().items(), key=lambda x: x[1]
        ):
            if tokidx >= self.tokenizer.vocab_size:
                self.added_tokens_list.append(tok)
                self.added_tokens_dict[tok] = tokidx
                self.added_tokens_ids.add(tokidx)
        self.specials = {
            tok: self.tokenizer.get_vocab()[tok]
            for tok in self.tokenizer.all_special_tokens
        }
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.vocab_size_base = self.tokenizer.vocab_size
        self.vocab_size = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def hf_tokens(self) -> Iterable[Tuple[bytes, float, TokenType]]:
        reverse_vocab = {
            id: encoded_tok for encoded_tok, id in self.tokenizer.get_vocab().items()
        }
        for token_id in range(self.vocab_size_base):
            if token_id in self.added_tokens_ids:
                continue
            token_text = reverse_vocab[token_id]
            yield (
                token_text,
                self.get_token_score(token_id),
                self.get_token_type(token_id, token_text, self.special_ids),
            )

    def get_token_type(
        self, token_id: int, token_text: bytes, special_ids: Set[int]
    ) -> TokenType:
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token_text):
            return TokenType.BYTE
        return TokenType.CONTROL if token_id in special_ids else TokenType.NORMAL

    def get_token_score(self, token_id: int) -> float:
        return -1000.0

    def added_tokens(self) -> Iterable[Tuple[bytes, float, TokenType]]:
        for text in self.added_tokens_list:
            if text in self.specials:
                toktype = self.get_token_type(self.specials[text], "", self.special_ids)
                score = self.get_token_score(self.specials[text])
            else:
                toktype = TokenType.USER_DEFINED
                score = -1000.0
            yield text, score, toktype

    def has_newline_token(self):
        return "<0x0A>" in self.tokenizer.vocab or "\n" in self.tokenizer.vocab

    def all_tokens(self) -> Iterable[Tuple[bytes, float, TokenType]]:
        yield from self.hf_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<HfVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

    @staticmethod
    def load(path: Path) -> "HfVocab":
        added_tokens_path = path.parent / "added_tokens.json"
        return HfVocab(path, added_tokens_path if added_tokens_path.exists() else None)


def translate_weight_names(name):
    name = name.replace("model.layers.", "blk.")
    name = name.replace("block_sparse_moe.gate", "ffn_gate_inp")
    pattern = r"block_sparse_moe\.experts\.(\d+)\.w1\.weight"
    replacement = r"ffn_gate.\1.weight"
    name = re.sub(pattern, replacement, name)
    pattern = r"block_sparse_moe\.experts\.(\d+)\.w2\.weight"
    replacement = r"ffn_down.\1.weight"
    name = re.sub(pattern, replacement, name)
    pattern = r"block_sparse_moe\.experts\.(\d+)\.w3\.weight"
    replacement = r"ffn_up.\1.weight"
    name = re.sub(pattern, replacement, name)

    name = name.replace("mlp.gate_proj", "ffn_gate")
    name = name.replace("mlp.down_proj", "ffn_down")
    name = name.replace("mlp.up_proj", "ffn_up")
    name = name.replace("self_attn.q_proj", "attn_q")
    name = name.replace("self_attn.k_proj", "attn_k")
    name = name.replace("self_attn.v_proj", "attn_v")
    name = name.replace("self_attn.o_proj", "attn_output")
    name = name.replace("input_layernorm", "attn_norm")
    name = name.replace("post_attention_layernorm", "ffn_norm")
    name = name.replace("model.embed_tokens", "token_embd")
    name = name.replace("model.norm", "output_norm")
    name = name.replace("lm_head", "output")
    return name


def permute_weights(weights, n_head, n_head_kv=None):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    reshaped = weights.reshape(
        n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
    )
    swapped = reshaped.transpose(1, 2)
    final_shape = weights.shape
    return swapped.reshape(final_shape)


def prepare_metadata(config, vocab):
    metadata = {
        "general.name": "llama",
        "llama.context_length": (
            torch.tensor(config["max_position_embeddings"], dtype=torch.uint32)
            if config.get("max_position_embeddings") is not None
            else None
        ),
        "llama.embedding_length": (
            torch.tensor(config["hidden_size"], dtype=torch.uint32)
            if config.get("hidden_size") is not None
            else None
        ),
        "llama.block_count": (
            torch.tensor(config["num_hidden_layers"], dtype=torch.uint32)
            if config.get("num_hidden_layers") is not None
            else None
        ),
        "llama.feed_forward_length": (
            torch.tensor(config["intermediate_size"], dtype=torch.uint32)
            if config.get("intermediate_size") is not None
            else None
        ),
        "llama.rope.dimension_count": (
            torch.tensor(
                config["hidden_size"] // config["num_attention_heads"],
                dtype=torch.uint32,
            )
            if config.get("hidden_size") is not None
            and config.get("num_attention_heads") is not None
            else None
        ),
        "llama.attention.head_count": (
            torch.tensor(config["num_attention_heads"], dtype=torch.uint32)
            if config.get("num_attention_heads") is not None
            else None
        ),
        "llama.attention.head_count_kv": (
            torch.tensor(
                config.get("num_key_value_heads", config["num_attention_heads"]),
                dtype=torch.uint32,
            )
            if config.get("num_attention_heads") is not None
            else None
        ),
        "llama.expert_count": (
            torch.tensor(config.get("num_local_experts", None), dtype=torch.uint32)
            if config.get("num_local_experts") is not None
            else None
        ),
        "llama.expert_used_count": (
            torch.tensor(config.get("num_experts_per_tok", None), dtype=torch.uint32)
            if config.get("num_experts_per_tok") is not None
            else None
        ),
        "llama.attention.layer_norm_rms_epsilon": (
            torch.tensor(config.get("rms_norm_eps", 1e-05))
            if config.get("rms_norm_eps") is not None
            else None
        ),
        "llama.rope.freq_base": (
            torch.tensor(config.get("rope_theta", 10000), dtype=torch.float32)
            if config.get("rope_theta") is not None
            else None
        ),
    }

    rope_scaling = config.get("rope_scaling")
    if rope_scaling is not None and (typ := rope_scaling.get("type")):
        rope_factor = rope_scaling.get("factor")
        f_rope_scale = rope_factor
        if typ == "linear":
            rope_scaling_type = "linear"
            metadata["llama.rope.scaling.type"] = rope_scaling_type
            metadata["llama.rope.scaling.factor"] = torch.tensor(f_rope_scale)

    metadata["general.file_type"] = torch.tensor(
        GGMLFileType.GGML_TYPE_F16.value, dtype=torch.uint32
    )
    metadata["general.quantization_version"] = torch.tensor(
        GGMLFileType.GGML_TYPE_F16.value, dtype=torch.uint32
    )
    metadata["general.name"] = config.get("_name_or_path", "llama").split("/")[-1]
    metadata["general.architecture"] = "llama"
    metadata["general.alignment"] = torch.tensor(32, dtype=torch.uint32)

    metadata["tokenizer.ggml.model"] = "llama"
    tokens = []
    scores = []
    toktypes = []
    for text, score, toktype in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype.value)
    assert len(tokens) == vocab.vocab_size
    metadata["tokenizer.ggml.tokens"] = tokens
    metadata["tokenizer.ggml.scores"] = torch.tensor(scores, dtype=torch.float32)
    metadata["tokenizer.ggml.token_type"] = torch.tensor(toktypes, dtype=torch.uint32)
    if vocab.tokenizer.bos_token_id is not None:
        metadata["tokenizer.ggml.bos_token_id"] = torch.tensor(
            vocab.tokenizer.bos_token_id, dtype=torch.uint32
        )
    if vocab.tokenizer.eos_token_id is not None:
        metadata["tokenizer.ggml.eos_token_id"] = torch.tensor(
            vocab.tokenizer.eos_token_id, dtype=torch.uint32
        )
    if vocab.tokenizer.unk_token_id is not None:
        metadata["tokenizer.ggml.unknown_token_id"] = torch.tensor(
            vocab.tokenizer.unk_token_id, dtype=torch.uint32
        )

    metadata = {k: v for k, v in metadata.items() if v is not None}
    return metadata


def convert_to_gguf(
    model_path: Union[str, Path],
    weights: dict,
    config: dict,
    output_file_path: str,
):
    if isinstance(model_path, str):
        model_path = Path(model_path)

    quantization = config.get("quantization", None)
    if quantization:
        raise NotImplementedError(
            "Conversion of quantized models is not yet supported."
        )
    print("Converting to GGUF format")
    weights = {
        k: (
            permute_weights(
                v, config["num_attention_heads"], config["num_attention_heads"]
            )
            if "self_attn.q_proj.weight" in k
            else (
                permute_weights(
                    v, config["num_attention_heads"], config["num_attention_heads"]
                )
                if "self_attn.k_proj.weight" in k
                else v
            )
        )
        for k, v in weights.items()
    }

    weights = {translate_weight_names(k): v for k, v in weights.items()}

    if not (model_path / "tokenizer.json").exists():
        raise ValueError("Tokenizer json not found")

    vocab = HfVocab.load(model_path)
    metadata = prepare_metadata(config, vocab)

    weights = {
        k: (
            v.to(torch.float32).to(torch.float16)
            if v.dtype == torch.bfloat16
            else v.to(torch.float32)
            if "norm" in k
            else v
        )
        for k, v in weights.items()
    }

    try:
        import gguf
    except ImportError:
        raise ImportError(
            "Install the gguf package to save in GGUF format: pip install gguf"
        )

    writer = gguf.GGUFWriter(output_file_path, "llama")

    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                if value.dtype in (torch.uint32, torch.int32, torch.int64):
                    writer.add_uint32(key, value.item())
                elif value.dtype == torch.float32:
                    writer.add_float32(key, value.item())
            else:
                writer.add_array(key, value.numpy())
        elif isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)

    for key, tensor in weights.items():
        writer.add_tensor(key, tensor.contiguous().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Converted GGUF model saved as: {output_file_path}")
