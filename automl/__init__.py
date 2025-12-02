"""
AutoML System - A fully automated machine learning pipeline
Handles all the nightmares of real-world data and produces production-ready models
"""

__version__ = "1.0.0"
__author__ = "AutoML Team"

from .data.ingestion import DataIngestor
from .data.profiling import DataProfiler
from .preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from .models.trainer import ModelTrainer
from .reports.report_generator import ReportGenerator

__all__ = [
    'DataIngestor',
    'DataProfiler', 
    'PreprocessingPipelineBuilder',
    'ModelTrainer',
    'ReportGenerator',
]
