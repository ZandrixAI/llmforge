# Memory System Guide

LLMForge includes a SQLite-based memory system for storing and recalling good responses.

## MemoryDB Class

```python
from llmforge.rl import MemoryDB

# Initialize memory database
memory = MemoryDB("llmforge_memory.db")
```

## Methods

### store()

Store a response with metadata:

```python
memory.store(
    prompt="What is Python?",
    response="Python is a high-level programming language...",
    strategy="refine",
    quality_score=0.8
)
```

### recall()

Find similar prompts:

```python
results = memory.recall("learn python", limit=5)

for result in results:
    print(result)
```

### get_best()

Get the best known response:

```python
best = memory.get_best("python tutorial")
if best:
    print(best)
```

## Keyword Extraction

The memory system uses keyword matching:

```python
from llmforge.rl import MemoryDB

memory = MemoryDB()

# Extract keywords from text
keywords = memory._extract_keywords("How do I learn Python programming?")
print(keywords)  # ['learn', 'python', 'programming']
```

### Stopwords

Common words are filtered out:

```
the, a, an, is, are, was, were, to, of, in,
for, on, with, at, by, from, as, and, or, but,
what, how, why, when, where, who
```

## Quality Scoring

```python
from llmforge.rl import QualityScorer

scorer = QualityScorer()

# Score a response
score = scorer.score("This is a well-written response.")
print(score)  # 0.5 - 1.0
```

### Scoring Criteria

| Criterion | Score |
|-----------|-------|
| Length 50-300 chars | +0.1 |
| Ends with punctuation | +0.1 |
| No filler words | +0.1 |
| Has structure markers | +0.1 |

## Database Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    strategy TEXT,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_prompt ON memories(prompt);
```