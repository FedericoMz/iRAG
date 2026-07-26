"""Shared visual styles for experiment trajectories."""

from __future__ import annotations


TRAJECTORY_STYLES = {
    "FEA": ("#2563eb", "-", 2.3),
    "Cumulative LLM gold coverage": ("#16a34a", "-", 2.1),
    "Cumulative human-reference coverage": ("#0891b2", "-", 1.8),
    "Cumulative final-decision error rate": ("#dc2626", "-", 1.8),
    "Cumulative human-only baseline error rate": ("#6b7280", "--", 1.8),
}
