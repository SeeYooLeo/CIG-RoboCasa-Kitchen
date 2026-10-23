import json
import logging
import random


class NegativeInstructionPool:
    def __init__(self, pool_path: str):
        with open(pool_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            self.instructions: list[str] = raw["instructions"]
            self._categories: dict[str, list[int]] = raw.get("categories", {})
        else:
            self.instructions = raw
            self._categories = {}
        self.N = len(self.instructions)
        self._inst_to_idx = {inst: i for i, inst in enumerate(self.instructions)}
        self._inst_to_cat: dict[int, str] = {}
        self._cat_indices: dict[str, list[int]] = {}
        for cat, idxs in self._categories.items():
            self._cat_indices[cat] = idxs
            for idx in idxs:
                self._inst_to_cat[idx] = cat
        logging.info(f"NegativeInstructionPool: {self.N} instructions from {pool_path}")
        if self._categories:
            summary = ", ".join(
                (f"{c}={len(ix)}" for c, ix in self._categories.items())
            )
            logging.info(f"NegativeInstructionPool categories: {summary}")

    def sample_negatives(
        self,
        positive_instructions: list[str],
        K: int,
        same_ratio: float = 0.0,
        cross_ratio: float = 0.0,
    ) -> list[list[str]]:
        B = len(positive_instructions)
        use_categories = (same_ratio > 0 or cross_ratio > 0) and bool(self._categories)
        n_same = int(K * same_ratio) if use_categories else 0
        n_cross = int(K * cross_ratio) if use_categories else 0
        neg_indices: list[list[int]] = [[] for _ in range(K)]
        for b in range(B):
            pos_idx = self._inst_to_idx.get(positive_instructions[b], -1)
            chosen: list[int] = []
            if use_categories:
                pos_cat = self._inst_to_cat.get(pos_idx)
                if pos_cat and pos_cat in self._cat_indices:
                    same_pool = [i for i in self._cat_indices[pos_cat] if i != pos_idx]
                    actual_same = min(n_same, len(same_pool))
                    if actual_same > 0:
                        chosen.extend(random.sample(same_pool, actual_same))
                if n_cross > 0:
                    if pos_cat and pos_cat != "pnp":
                        other_cats = [
                            c for c in self._cat_indices if c != pos_cat and c != "pnp"
                        ]
                    else:
                        other_cats = [c for c in self._cat_indices if c != pos_cat]
                    cross_pool = []
                    for c in other_cats:
                        cross_pool.extend(self._cat_indices[c])
                    cross_pool = [
                        i for i in cross_pool if i != pos_idx and i not in chosen
                    ]
                    actual_cross = min(n_cross, len(cross_pool))
                    if actual_cross > 0:
                        chosen.extend(random.sample(cross_pool, actual_cross))
            n_fill = K - len(chosen)
            if n_fill > 0:
                remaining = [
                    i for i in range(self.N) if i != pos_idx and i not in chosen
                ]
                chosen.extend(random.sample(remaining, min(n_fill, len(remaining))))
            while len(chosen) < K:
                chosen.append(random.randint(0, self.N - 1))
            for k in range(K):
                neg_indices[k].append(chosen[k])
        return [[self.instructions[i] for i in slot] for slot in neg_indices]
