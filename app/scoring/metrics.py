from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    hit = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return hit / len(y_true)


def weighted_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    support = Counter(y_true)
    total = len(y_true)
    if total == 0:
        return 0.0
    weighted_sum = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        weighted_sum += f1 * support.get(label, 0)
    return weighted_sum / total


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    values: List[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        values.append(f1)
    return sum(values) / len(values)


def multilabel_f1(gold: Iterable[str], pred: Iterable[str]) -> float:
    g: Set[str] = set(gold)
    p: Set[str] = set(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    inter = len(g & p)
    precision = inter / len(p) if p else 0.0
    recall = inter / len(g) if g else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

