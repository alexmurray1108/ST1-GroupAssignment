# Manual Testing Evidence

This document records manual checks performed against the console application and prediction workflow in the current codebase.

| Scenario | Input | Expected Result | Actual Result | Evidence |
|---|---|---|---|---|
| Missing model file | Predict before training | Friendly error message shown | Passed - the application stopped the prediction flow and showed a user-friendly message when the trained model was missing. | Screenshot 1 |
| Invalid image path | wrong_file.jpg | File cannot be read | Passed - the application returned a friendly error message for an unreadable or missing image path. | Screenshot 2 |
| Invalid menu option | 9 | Program asks again | Passed - the console displayed "Invalid option. Please select again." and returned to the menu. | Screenshot 3 |
| Valid prediction | sample image | Predicted class displayed | Passed - a predicted macroinvertebrate class was printed to the console. | Screenshot 4 |

Additional notes:

- The menu-driven application is implemented in [src/console_app.py](src/console_app.py).
- Image loading and validation are handled by [src/services/Image_processor.py](src/services/Image_processor.py).
- Prediction requires a trained model saved in [outputs/models/macro_classifier.joblib](outputs/models/macro_classifier.joblib).
- The planning PDF also recorded that invalid image file paths previously caused the program to break, so manual testing should confirm the application now shows a friendly error instead of crashing.
- The classifier service prints progress messages for each stage of training and error handling, which makes it easier to verify that each step completes successfully.
