"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (27/04/2026)
Group Assignment
*******************************
"""

from pathlib import Path
import pandas as pd

from src.config import EDA_OUTPUT_DIR
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService, save_sample_grid

class WorkflowService:
    """Workflow for Stage 1: indexing, summary, and EDA generation"""

    def __init__(self) -> None:
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.indexer = DatasetIndexer()
        self.dataframe:pd.DataFrame | None = None

    def load_dataframe(self) -> pd.DataFrame:
        """Load and save the indexed dataset"""
        if self.dataframe is None:
            self.dataframe = self.indexer.build_dataframe()
        return self.dataframe
    
    def show_summary(self) -> dict[str, float]:
        """Print summary"""
        df = self.load_dataframe()
        eda = EDAService(df, EDA_OUTPUT_DIR)
        summary = eda.build_summary()
        print(summary)
        return summary
    
    def generate_eda(self) -> None:
        """Generate and save outputs"""
        df = self.load_dataframe()
        eda = EDAService(df, EDA_OUTPUT_DIR)
        eda.save_class_distribution()
        eda.save_image_size_distribution()
        save_sample_grid(df, EDA_OUTPUT_DIR / "sample_grid.png")