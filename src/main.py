"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (30/04/2026)
Group Assignment

NOTE: Code is adapted from the Assignment 3 Full Guidance, 
with some modifications to better fit the needs of this project.
*******************************
"""

import sys
from pathlib import Path

# Enable direct script execution by adding project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.workflow_service import WorkflowService
from console_app import ConsoleApp

def main() -> None:
    """Run GUI"""
    workflow_service = WorkflowService()
    app = ConsoleApp(workflow_service)
    app.run()


if __name__ == "__main__":
    main()