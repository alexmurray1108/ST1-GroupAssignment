"""
*******************************
Assessment: 3
Author: u3281627
Group Members: u3294093, u3271260
Date: 28/04/2026 (TODAYS DATE)
Group Assignment


NOTE: Code is adapted from the Assignment 3 Full Guidance,
with some modifications to better fit the needs of this project.
*******************************
"""
from services.workflow_service import WorkflowService

class ConsoleApp:
    """
    Menu-driven console application
    """
    def __init__(self, workflow_service: "WorkflowService") -> None:
        self.workflow_service = workflow_service
    
    def run(self):
        """
        Start menu until user exits
        """

        while True:
            print("\Macoinvertebrate Image Analysis & Processing System")
            print("1. Show dataset summary")
            print("2. Generate EDA outputs")
            print("3. Train the baseline classifier")
            print("4. Predictive analysis of an image")
            print("5. Exit")

            choice = input("Please select an option: ").strip()

            if choice == "1":
                self.workflow_service.show_summary()
            elif choice == "2":
                self.workflow_service.generate_eda()
            elif choice == "3":
                self.workflow_service.train_classifier()
            elif choice == "4":
                image_path = input ("Enter image path: ").strip()
                self.workflow_service.predict_image(image_path)
            elif choice == "5":
                print("Exiting application")
                break
            else:
                print("Invalid option. Please select again.")


