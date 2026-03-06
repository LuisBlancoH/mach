"""
Hippocampus: episodic memory with similarity-based retrieval.

Stores experiences as (embedding, text) pairs. Retrieves by cosine
similarity to current state. Writes are gated by surprise (|TD error|).
Memories decay over time unless reinforced.

Persists to disk — survives restarts.

Brain mapping:
- Encoding: activation summary → embedding (entorhinal cortex → hippocampus)
- Storage: content-addressable (CA3 pattern completion)
- Retrieval: similarity search (CA3 → CA1 → cortex)
- Consolidation: replay → patch updates (hippocampus → cortex during sleep)
- Forgetting: decay without reinforcement
"""

import json
import os
import torch
import numpy as np


class Hippocampus:
    """Episodic memory store with embedding-based retrieval."""

    def __init__(self, embedding_dim, max_memories=1000, surprise_threshold=0.5,
                 decay_rate=0.999, save_path=None):
        """
        Args:
            embedding_dim: dimension of activation summary embeddings
            max_memories: maximum stored memories (FIFO when full)
            surprise_threshold: minimum |TD error| to store a memory
            decay_rate: per-step strength decay (0.999 = ~1000 step half-life)
            save_path: path to persist memories to disk (None = no persistence)
        """
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self.surprise_threshold = surprise_threshold
        self.decay_rate = decay_rate
        self.save_path = save_path

        # Storage: parallel lists
        self.embeddings = []    # list of numpy arrays (embedding_dim,)
        self.texts = []         # list of strings (the experience)
        self.strengths = []     # list of floats (decays over time, reinforced on retrieval)
        self.td_errors = []     # list of floats (surprise at storage time)

        # Load from disk if exists
        if save_path and os.path.exists(save_path):
            self.load(save_path)

    def store(self, embedding, text, td_error):
        """Store an experience if surprising enough.

        Args:
            embedding: activation summary tensor (embedding_dim,)
            text: the experience text (e.g. "45 ? 23 = 68")
            td_error: scalar surprise magnitude
        """
        if abs(td_error) < self.surprise_threshold:
            return False

        emb_np = embedding.detach().cpu().numpy().astype(np.float32)

        # Check for near-duplicate (don't store identical experiences)
        if self.embeddings:
            sims = self._cosine_similarities(emb_np)
            if sims.max() > 0.95:
                # Reinforce existing memory instead
                idx = sims.argmax()
                self.strengths[idx] = min(self.strengths[idx] + 0.1, 1.0)
                return False

        self.embeddings.append(emb_np)
        self.texts.append(text)
        self.strengths.append(1.0)
        self.td_errors.append(abs(td_error))

        # Evict weakest if over capacity
        if len(self.embeddings) > self.max_memories:
            weakest = int(np.argmin(self.strengths))
            self.embeddings.pop(weakest)
            self.texts.pop(weakest)
            self.strengths.pop(weakest)
            self.td_errors.pop(weakest)

        return True

    def retrieve(self, embedding, top_k=5):
        """Retrieve most similar memories.

        Args:
            embedding: current activation summary tensor (embedding_dim,)
            top_k: number of memories to retrieve

        Returns:
            list of (text, similarity, strength) tuples, sorted by similarity
        """
        if not self.embeddings:
            return []

        emb_np = embedding.detach().cpu().numpy().astype(np.float32)
        sims = self._cosine_similarities(emb_np)

        # Weight by strength (decayed memories are less retrievable)
        weighted_sims = sims * np.array(self.strengths)

        # Top-k
        k = min(top_k, len(self.embeddings))
        top_indices = np.argsort(weighted_sims)[-k:][::-1]

        results = []
        for idx in top_indices:
            if weighted_sims[idx] > 0.0:
                # Reinforce retrieved memories (retrieval strengthens memory)
                self.strengths[idx] = min(self.strengths[idx] + 0.05, 1.0)
                results.append((self.texts[idx], float(sims[idx]), self.strengths[idx]))

        return results

    def decay_all(self):
        """Apply passive decay to all memories. Call periodically."""
        for i in range(len(self.strengths)):
            self.strengths[i] *= self.decay_rate

        # Remove dead memories (strength < 0.01)
        alive = [i for i, s in enumerate(self.strengths) if s >= 0.01]
        if len(alive) < len(self.strengths):
            self.embeddings = [self.embeddings[i] for i in alive]
            self.texts = [self.texts[i] for i in alive]
            self.strengths = [self.strengths[i] for i in alive]
            self.td_errors = [self.td_errors[i] for i in alive]

    def get_replay_batch(self, batch_size=10):
        """Sample memories for consolidation replay, weighted by strength.

        Returns:
            list of (text, td_error) tuples
        """
        if not self.texts:
            return []

        strengths = np.array(self.strengths)
        probs = strengths / strengths.sum()
        k = min(batch_size, len(self.texts))
        indices = np.random.choice(len(self.texts), size=k, replace=False, p=probs)

        return [(self.texts[i], self.td_errors[i]) for i in indices]

    def _cosine_similarities(self, query):
        """Compute cosine similarity between query and all stored embeddings."""
        if not self.embeddings:
            return np.array([])

        matrix = np.stack(self.embeddings)  # (n_memories, embedding_dim)
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return matrix_norms @ query_norm  # (n_memories,)

    def save(self, path=None):
        """Persist memories to disk."""
        path = path or self.save_path
        if path is None:
            return

        data = {
            'embeddings': [e.tolist() for e in self.embeddings],
            'texts': self.texts,
            'strengths': self.strengths,
            'td_errors': self.td_errors,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path=None):
        """Load memories from disk."""
        path = path or self.save_path
        if path is None or not os.path.exists(path):
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.embeddings = [np.array(e, dtype=np.float32) for e in data['embeddings']]
        self.texts = data['texts']
        self.strengths = data['strengths']
        self.td_errors = data['td_errors']

    def __len__(self):
        return len(self.texts)

    def __repr__(self):
        return f"Hippocampus({len(self)} memories, dim={self.embedding_dim})"
