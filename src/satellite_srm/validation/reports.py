"""Automated validation report generator outputting JSON, CSV, and formatted HTML summaries."""
import os
import json
import pandas as pd
from typing import Dict, Any

class ValidationReportGenerator:
    """Formats benchmark results into professional, shareable validation reports."""

    @staticmethod
    def generate(eval_results: Dict[str, Any], output_dir: str = "outputs/reports"):
        os.makedirs(output_dir, exist_ok=True)
        metrics = eval_results.get("metrics", {})

        # 1. JSON Report
        json_path = os.path.join(output_dir, "validation_report.json")
        with open(json_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # 2. CSV Report
        rows = []
        for metric_name, values in metrics.items():
            rows.append({
                "Metric": metric_name,
                "Bicubic": values.get("Bicubic", 0.0),
                "SR_Model": values.get("SR_Model", 0.0),
                "Gain": values.get("Gain", 0.0)
            })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "baseline_vs_model.csv")
        df.to_csv(csv_path, index=False)

        # 3. HTML Report
        html_path = os.path.join(output_dir, "validation_report.html")
        table_html = df.to_html(classes="table table-striped", index=False)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Validation Report - Satellite-SRM</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f4f6f8; color: #333; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        h1 {{ color: #1e3a8a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; border: 1px solid #cbd5e1; text-align: left; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        .footer {{ margin-top: 30px; font-size: 0.85em; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Satellite-SRM: Quantitative Model Validation Report</h1>
        <p><strong>Problem Statement:</strong> 26142 (NTRO) | <strong>Model:</strong> Multispectral SwinIR</p>
        <p><strong>Target Resolution:</strong> 10m Sentinel-2 L2A &rarr; sub-4m Enhanced Product</p>
        <h3>Benchmark vs. Bicubic Baseline</h3>
        {table_html}
        <div class="footer">
            <p>Generated automatically by Satellite-SRM Validation Suite.</p>
        </div>
    </div>
</body>
</html>"""
        with open(html_path, "w") as f:
            f.write(html_content)

        return json_path, csv_path, html_path
