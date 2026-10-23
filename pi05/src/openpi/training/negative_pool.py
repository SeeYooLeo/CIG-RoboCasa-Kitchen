import json
import logging
import random
import torch


class NegativeInstructionPool:
    def __init__(self, pool_path: str, tokenizer, device):
        with open(pool_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            self.instructions = raw["instructions"]
            self._categories = raw.get("categories", {})
        else:
            self.instructions = raw
            self._categories = {}
        logging.info(
            f"NegativeInstructionPool: loaded {len(self.instructions)} instructions from {pool_path}"
        )
        self._inst_to_idx = {inst: i for i, inst in enumerate(self.instructions)}
        self._inst_to_cat = {}
        self._cat_indices = {}
        for cat, indices in self._categories.items():
            self._cat_indices[cat] = indices
            for idx in indices:
                self._inst_to_cat[idx] = cat
        if self._categories:
            cat_summary = ", ".join(
                (f"{c}={len(idxs)}" for c, idxs in self._categories.items())
            )
            logging.info(f"NegativeInstructionPool: categories: {cat_summary}")
        all_tokens = []
        all_masks = []
        for inst in self.instructions:
            tokens, masks = tokenizer.tokenize(inst)
            all_tokens.append(torch.from_numpy(tokens))
            all_masks.append(torch.from_numpy(masks))
        self.tokens = torch.stack(all_tokens).to(device)
        self.masks = torch.stack(all_masks).to(device)
        self.N = len(self.instructions)
        logging.info(
            f"NegativeInstructionPool: pre-tokenized {self.N} instructions, tokens shape={self.tokens.shape}"
        )

    def sample_negatives(
        self,
        positive_instructions: list[str],
        K: int,
        same_ratio: float = 0.0,
        cross_ratio: float = 0.0,
    ):
        B = len(positive_instructions)
        use_categories = (same_ratio > 0 or cross_ratio > 0) and bool(self._categories)
        n_same = int(K * same_ratio) if use_categories else 0
        n_cross = int(K * cross_ratio) if use_categories else 0
        neg_indices = [[] for _ in range(K)]
        for b in range(B):
            pos_idx = self._inst_to_idx.get(positive_instructions[b], -1)
            chosen = []
            if use_categories:
                pos_cat = self._inst_to_cat.get(pos_idx)
                if pos_cat and pos_cat in self._cat_indices:
                    cat_pool = [i for i in self._cat_indices[pos_cat] if i != pos_idx]
                    actual_same = min(n_same, len(cat_pool))
                    if actual_same > 0:
                        chosen.extend(random.sample(cat_pool, actual_same))
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
                all_remaining = [
                    i for i in range(self.N) if i != pos_idx and i not in chosen
                ]
                chosen.extend(
                    random.sample(all_remaining, min(n_fill, len(all_remaining)))
                )
            while len(chosen) < K:
                chosen.append(random.randint(0, self.N - 1))
            for k in range(K):
                neg_indices[k].append(chosen[k])
        neg_tokens = []
        neg_masks = []
        for k in range(K):
            indices = torch.tensor(neg_indices[k], dtype=torch.long)
            neg_tokens.append(self.tokens[indices])
            neg_masks.append(self.masks[indices])
        return (neg_tokens, neg_masks)
