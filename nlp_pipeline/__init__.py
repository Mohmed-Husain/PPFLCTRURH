"""
nlp_pipeline — Modular NLP engine for Privacy-Preserving Federated
Hierarchical Multi-Label Clinical Text Classification (AAAI-27).

Public API:
    - FederatedNLPClient   : FL-ready client wrapper
    - ClinicalHMLTCModel   : PubMedBERT-based classifier
    - PipelineConfig       : Centralized configuration
    - HierarchicalLabelEncoder : Multi-hot encoding with hierarchy
"""

from nlp_pipeline.configs.config import PipelineConfig
from nlp_pipeline.api.interface import FederatedNLPClient

__version__ = "0.1.0"
__all__ = [
    "PipelineConfig",
    "FederatedNLPClient",
]
