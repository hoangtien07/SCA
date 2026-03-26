from .base_collector import BaseCollector, Paper
from .semantic_scholar import SemanticScholarCollector
from .pubmed import PubMedCollector
from .pmc_oa import PMCOpenAccessCollector

__all__ = [
    "BaseCollector",
    "Paper",
    "SemanticScholarCollector",
    "PubMedCollector",
    "PMCOpenAccessCollector",
]
