import os
import sys

import great_expectations as gx
import pandas as pd
from great_expectations.dataset import PandasDataset


def validate_data():
    # Читаем данные
    df = pd.read_csv("data/raw/tips.csv")

    # Создаем GX DataSet
    ge_df = PandasDataset(df)

    # Создаем expectations
    ge_df.expect_column_values_to_not_be_null("total_bill")
    ge_df.expect_column_values_to_not_be_null("tip")
    ge_df.expect_column_values_to_not_be_null("size")
    ge_df.expect_column_values_to_be_between("total_bill", min_value=0, max_value=100)
    ge_df.expect_column_values_to_be_between("size", min_value=1, max_value=10)

    # Получаем expectation suite
    expectation_suite = ge_df.get_expectation_suite()
    expectation_suite.expectation_suite_name = "tips_validation_suite"

    # Создаем context и добавляем suite
    try:
        context = gx.get_context()
    except Exception:
        # Инициализируем GX в текущей директории
        context = gx.data_context.FileDataContext.create(".")

    # Добавляем или обновляем suite
    try:
        context.save_expectation_suite(expectation_suite)
    except Exception:
        context.add_expectation_suite(expectation_suite)

    # Валидируем данные
    validation_results = ge_df.validate()

    # Создаем директории для отчетов
    os.makedirs("reports/validation", exist_ok=True)

    # Генерируем простой HTML отчет с результатами валидации
    success_rate = (
        validation_results.statistics["successful_expectations"]
        / validation_results.statistics["evaluated_expectations"]
    )

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tips Dataset Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 8px; }}
        .success {{ color: #2e7d32; }}
        .failed {{ color: #d32f2f; }}
        .summary {{ margin: 20px 0; }}
        .expectation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #2196f3; }}
        .expectation.success {{ border-color: #4caf50; background: #f1f8e9; }}
        .expectation.failed {{ border-color: #f44336; background: #ffebee; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Tips Dataset Validation Report</h1>
        <div class="summary">
            <p><strong>Success Rate:</strong> <span class="{'success' if success_rate == 1.0 else 'failed'}">{success_rate:.1%}</span></p>
            <p><strong>Total Expectations:</strong> {validation_results.statistics["evaluated_expectations"]}</p>
            <p><strong>Successful:</strong> <span class="success">{validation_results.statistics["successful_expectations"]}</span></p>
            <p><strong>Failed:</strong> <span class="failed">{validation_results.statistics["unsuccessful_expectations"]}</span></p>
        </div>
    </div>

    <h2>Expectation Results:</h2>
"""

    for result in validation_results.results:
        status = "success" if result.success else "failed"
        status_icon = "✅" if result.success else "❌"
        expectation_type = result.expectation_config.expectation_type
        column = result.expectation_config.kwargs.get("column", "")

        html_content += f"""
    <div class="expectation {status}">
        <strong>{status_icon} {expectation_type}</strong>
        {f'<br><em>Column:</em> {column}' if column else ''}
        <br><em>Success:</em> {result.success}
    </div>"""

    html_content += """
    <div style="margin-top: 40px; padding: 20px; background: #e3f2fd; border-radius: 8px;">
        <h3>Dataset Summary</h3>
        <p><strong>Shape:</strong> {rows} rows × {cols} columns</p>
        <p><strong>Generated:</strong> {timestamp}</p>
    </div>
</body>
</html>""".format(
        rows=len(df),
        cols=len(df.columns),
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Сохраняем HTML отчет
    with open("reports/validation/index.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Validation report generated at: reports/validation/index.html")

    # Проверяем результат валидации
    if not validation_results.success:
        print("Data validation failed!")
        sys.exit(1)

    print("Data validation passed!")


if __name__ == "__main__":
    validate_data()
