# Manual Testing Evidence

This document records manual checks performed against the console application and prediction workflow in the current codebase.

| Scenario | Input | Expected Result | Actual Result | Evidence |
|---|---|---|---|---|
| Invalid image path | wrong_file.jpg | File cannot be read | Passed - the application returned a friendly error message for an unreadable or missing image path. | <img src="Screenshot%201.png" width="300" /> |
| Invalid menu option | 9 | Program asks again | Passed - the console displayed "Invalid option. Please select again." and returned to the menu. | <img src="Screenshot%202.png" width="300" /> |
| Valid prediction | sample image | Predicted class displayed | Passed - a predicted macroinvertebrate class was printed to the console. | <img src="Screenshot%203.png" width="300" /> |