"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (30/04/2026)
Group Assignment
*******************************
"""

import sys
from pathlib import Path

# Enable direct script execution by adding project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.workflow_service import WorkflowService

def main() -> None:
    """Run Stage 1"""
    workflow = WorkflowService()
    workflow.show_summary()
    workflow.generate_eda()


if __name__ == "__main__":
    main()