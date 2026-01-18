"""
LLM GPU Scheduling Sub-Project

Extension of the heterogeneous GPU cluster scheduling simulator
for LLM inference workloads with continuous batching and token-level scheduling.

Optimization Objective: Minimize weighted waiting time
    Σ weight × (completion_time - arrival_time)
"""

__version__ = "0.1.0"
