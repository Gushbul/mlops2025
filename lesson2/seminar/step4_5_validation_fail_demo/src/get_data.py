import os
import pandas as pd


def create_bad_data():
    """Creates dataset with validation violations to demonstrate fail-fast"""
    os.makedirs("data/raw", exist_ok=True)

    # Create data that violates GE expectations
    bad_data = pd.DataFrame(
        {
            "total_bill": [
                15.0,
                None,
                150.0,
                25.0,
                30.0,
            ],  # Contains null and value > 100
            "tip": [2.0, 3.0, None, 4.0, 5.0],  # Contains null
            "size": [0, 2, 15, 3, 4],  # Contains 0 (< 1) and 15 (> 10)
            "sex": ["Male", "Female", "Male", "Female", "Male"],
            "smoker": ["No", "Yes", "No", "Yes", "No"],
            "day": ["Sat", "Sun", "Sat", "Sun", "Sat"],
            "time": ["Dinner", "Dinner", "Dinner", "Dinner", "Dinner"],
        }
    )

    bad_data.to_csv("data/raw/tips.csv", index=False)
    print("Created dataset with validation violations:")
    print("- total_bill: null values and 150.0 > 100")
    print("- tip: null values")
    print("- size: 0 < 1 and 15 > 10")


if __name__ == "__main__":
    create_bad_data()
