"""
LLMForge - RL Engine with SQLite Memory
Uses SQLite for persistent memory storage.
"""

import re
import sqlite3
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

from llmforge.inference import InferenceEngine


class QualityScorer:
    """Simple quality scoring."""
    
    def score(self, text: str) -> float:
        if not text or len(text.strip()) < 10:
            return 0.0
        
        score = 0.5
        if 50 <= len(text) <= 300:
            score += 0.1
        if text.rstrip().endswith(('.', '!', '?')):
            score += 0.1
        fillers = [' um ', ' uh ', ' like ', ' you know ']
        if not any(f in text.lower() for f in fillers):
            score += 0.1
        if any(c in text for c in ['\n', '. ', '- ', '1. ']):
            score += 0.1
        return min(1.0, score)


class MemoryDB:
    """SQLite-based memory for storing responses."""
    
    def __init__(self, db_path: str = "llmforge_memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                strategy TEXT,
                quality_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt ON memories(prompt)
        """)
        self.conn.commit()
    
    def store(self, prompt: str, response: str, strategy: str, quality_score: float):
        self.conn.execute(
            "INSERT INTO memories (prompt, response, strategy, quality_score) VALUES (?, ?, ?, ?)",
            (prompt, response, strategy, quality_score)
        )
        self.conn.commit()
    
    def recall(self, prompt: str, limit: int = 3) -> List[Dict]:
        """Find similar prompts using keyword matching."""
        # Extract keywords
        keywords = self._extract_keywords(prompt)
        
        if not keywords:
            return []
        
        # Search for memories with matching keywords
        placeholders = ','.join(['?'] * len(keywords))
        query = f"""
            SELECT prompt, response, strategy, quality_score 
            FROM memories 
            WHERE {" OR ".join([f"prompt LIKE ?"] * len(keywords))}
            ORDER BY quality_score DESC
            LIMIT ?
        """
        like_patterns = [f"%{k}%" for k in keywords]
        cursor = self.conn.execute(query, like_patterns + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'prompt': row[0],
                'response': row[1],
                'strategy': row[2],
                'quality_score': row[3]
            })
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 
                    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'and', 'or', 'but',
                    'what', 'how', 'why', 'when', 'where', 'who'}
        words = re.findall(r'[a-zA-Z]{4,}', text.lower())
        keywords = [w for w in words if w not in stopwords]
        return keywords[:5]  # Top 5 keywords
    
    def get_best(self, prompt: str) -> Optional[str]:
        """Get best known response for a prompt."""
        keywords = self._extract_keywords(prompt)
        if not keywords:
            return None
        
        like_patterns = [f"%{k}%" for k in keywords]
        query = f"""
            SELECT response FROM memories 
            WHERE {" OR ".join([f"prompt LIKE ?"] * len(keywords))}
            ORDER BY quality_score DESC
            LIMIT 1
        """
        cursor = self.conn.execute(query, like_patterns)
        result = cursor.fetchone()
        return result[0] if result else None
    
    def close(self):
        self.conn.close()


class RLEngine(InferenceEngine):
    """
    RL-enhanced engine with persistent memory using SQLite.
    Stores good responses and recalls them for similar prompts.
    """
    
    def __init__(self, model_name: str, device=None, bits=16, offload=False, memory_db: str = None):
        super().__init__(model_name, device, bits=bits, offload=offload)
        self.scorer = QualityScorer()
        self.memory = MemoryDB(memory_db) if memory_db else MemoryDB()
        self.stats = {"generations": 0, "memory_hits": 0}
    
    def generate(self, prompt: str, max_tokens: int = 128, strategy: str = "auto") -> str:
        """Generate with RL self-improvement and memory."""
        self.stats["generations"] += 1
        
        # Check memory first
        known = self.memory.get_best(prompt)
        if known:
            self.stats["memory_hits"] += 1
            print(f"[Memory hit!]", end=" ")
            return known
        
        # Auto strategy selection
        if strategy == "auto":
            strategy = self._select_strategy(prompt)
        
        # Generate
        if strategy == "best_of_n":
            result = self._best_of_n(prompt, max_tokens, n=3)
        elif strategy == "refine":
            result = self._refine(prompt, max_tokens)
        elif strategy == "expand":
            result = self._expand(prompt, max_tokens)
        elif strategy == "consensus":
            result = self._consensus(prompt, max_tokens)
        else:
            result = super().generate(prompt, max_tokens)
        
        # Store in memory if quality is good
        score = self.scorer.score(result)
        if score >= 0.5:
            self.memory.store(prompt, result, strategy, score)
        
        return result
    
    def _select_strategy(self, prompt: str) -> str:
        p = prompt.lower()
        if len(p) < 20:
            return "expand"
        if any(w in p for w in ['explain', 'describe', 'what is']):
            return "refine"
        if any(w in p for w in ['create', 'write', 'list', 'make']):
            return "expand"
        return "refine"
    
    def _best_of_n(self, prompt: str, max_tokens: int, n: int = 3) -> str:
        candidates = []
        for i in range(n):
            temp = 0.3 + (i * 0.3)
            result = super().generate(prompt, max_tokens)
            candidates.append(result)
        return max(candidates, key=lambda x: self.scorer.score(x))
    
    def _refine(self, prompt: str, max_tokens: int) -> str:
        result = super().generate(prompt, max_tokens)
        score = self.scorer.score(result)
        
        if score < 0.6:
            improve_prompt = f"Task: {prompt}\n\nResponse: {result}\n\nImprove:"
            result = super().generate(improve_prompt, max_tokens)
        
        return result
    
    def _expand(self, prompt: str, max_tokens: int) -> str:
        expand_prompt = f"Detailed response to: {prompt}\n\nInclude:\n- Clear explanation\n- Key points"
        return super().generate(expand_prompt, max_tokens)
    
    def _consensus(self, prompt: str, max_tokens: int) -> str:
        candidates = [super().generate(prompt, max_tokens) for _ in range(3)]
        return max(candidates, key=lambda x: self.scorer.score(x))
    
    def get_stats(self) -> Dict:
        return self.stats
    
    def search_memory(self, prompt: str, limit: int = 3) -> List[Dict]:
        """Search memory for similar prompts."""
        return self.memory.recall(prompt, limit)
    
    def close(self):
        self.memory.close()