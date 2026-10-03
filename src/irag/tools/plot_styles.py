"""Shared visual styles for experiment trajectories."""

from __future__ import annotations


TRAJECTORY_STYLES = {
    "FEA": ("#2563eb", "-", 2.3),
    "Cumulative final-decision error rate": ("#dc2626", "-", 1.8),
    "Cumulative human-only baseline error rate": ("#6b7280", "--", 1.8),
    "Cumulative model-first replay error rate": ("#0f766e", "-.", 1.8),
    "Controller-free RAG-with-defer error": ("#0f766e", "-.", 1.8),
}
