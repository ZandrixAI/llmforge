"""
LLMForge - Clean Inference Engine
Works on CPU, GPU (CUDA), MPS (Apple Silicon), and TPU.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from llmforge.base_engine import BaseEngine


class InferenceEngine(BaseEngine):
    """Inference engine with generation and chat support."""

    def __init__(self, model_name, device=None, tensor_parallel=True, bits=16, offload=False):
        super().__init__(model_name, device, tensor_parallel, bits, offload)
        self.eos_token_ids = self._get_eos_ids()

    def _sample_token(self, logits, temperature=1.0):
        if temperature <= 0:
            return torch.argmax(logits).item()
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """Plain text generation."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        current_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        
        results = []
        for step in range(max_tokens):
            logits = self._forward(current_ids)
            next_logits = logits[0, -1, :] if len(logits.shape) == 3 else logits.squeeze()
            
            token_id = self._sample_token(next_logits, temperature)
            if token_id in self.eos_token_ids:
                break
            
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            results.append(text)
            
            current_ids = torch.cat([current_ids, torch.tensor([[token_id]]).to(self.device)], dim=1)
        
        return "".join(results)

    def chat(self, messages: List[dict], max_tokens: int = 128, temperature: float = 0.7, enable_thinking: bool = False) -> str:
        """Chat format generation."""
        chat_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        current_ids = torch.tensor(self.tokenizer.encode(chat_prompt, add_special_tokens=True)).unsqueeze(0).to(self.device)
        
        results = []
        for step in range(max_tokens):
            logits = self._forward(current_ids)
            next_logits = logits[0, -1, :] if len(logits.shape) == 3 else logits.squeeze()
            
            token_id = self._sample_token(next_logits, temperature)
            if token_id in self.eos_token_ids:
                break
            
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            results.append(text)
            
            current_ids = torch.cat([current_ids, torch.tensor([[token_id]]).to(self.device)], dim=1)
        
        return "".join(results)