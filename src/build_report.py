"""Build the assessment PDF from recorded results, without running training.

The report content is supplied through artifacts/report_data.json. All numerical
claims are taken from that file; the builder never fabricates validation scores.
Run ``python -m src.build_report --help`` for paths and rendering options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
INK = colors.HexColor("#15313C")
TEAL = colors.HexColor("#086779")
MUTED = colors.HexColor("#526973")
LIGHT = colors.HexColor("#ECF3F5")
RULE = colors.HexColor("#CFDFE4")
AMBER = colors.HexColor("#AC6B1A")
PAGE_WIDTH, PAGE_HEIGHT = A4
WIDTH = PAGE_WIDTH - 92


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    definitions = {
        "body": dict(fontName="Helvetica", fontSize=9.6, leading=14.1, textColor=INK, spaceAfter=8),
        "small": dict(fontName="Helvetica", fontSize=8.1, leading=11.5, textColor=MUTED, spaceAfter=6),
        "title": dict(fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=INK, spaceAfter=16),
        "subtitle": dict(fontName="Helvetica", fontSize=12.2, leading=18, textColor=MUTED, spaceAfter=18),
        "h1": dict(fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=INK, spaceAfter=13),
        "h2": dict(fontName="Helvetica-Bold", fontSize=11.1, leading=15, textColor=TEAL, spaceBefore=11, spaceAfter=7),
        "eyebrow": dict(fontName="Helvetica-Bold", fontSize=8.4, leading=12, textColor=TEAL, spaceAfter=9),
        "cell": dict(fontName="Helvetica", fontSize=8.6, leading=12, textColor=INK),
        "cell_head": dict(fontName="Helvetica-Bold", fontSize=8.6, leading=12, textColor=colors.white),
        "metric_value": dict(fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=TEAL, alignment=TA_CENTER),
        "metric_label": dict(fontName="Helvetica", fontSize=8.4, leading=12, textColor=MUTED, alignment=TA_CENTER),
        "code": dict(fontName="Courier", fontSize=7.8, leading=11.5, textColor=INK, spaceAfter=7),
    }
    return {name: ParagraphStyle(name, parent=base["Normal"], **props) for name, props in definitions.items()}


STYLES = make_styles()


def p(value: object, style: str = "body") -> Paragraph:
    # Only the builder supplies markup. Content read from JSON is always escaped.
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), STYLES[style])


def paragraphs(values: list[str], style: str = "body") -> list[Paragraph]:
    return [p(value, style) for value in values]


def section(label: str, title: str, introduction: str | None = None) -> list:
    result = [p(label.upper(), "eyebrow"), p(title, "h1")]
    if introduction:
        result.append(p(introduction))
    return result


def data_table(headers: list[str], rows: list[list], widths: list[float] | None = None) -> Table:
    content = [[p(x, "cell_head") for x in headers]]
    content.extend([[p(x, "cell") for x in row] for row in rows])
    result = Table(content, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
    ]))
    return result


def cards(items: list[dict]) -> Table:
    cells = [[p(item["value"], "metric_value"), Spacer(1, 5), p(item["label"], "metric_label")] for item in items]
    result = Table([cells], colWidths=[WIDTH / len(cells)] * len(cells))
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 13),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 13),
    ]))
    return result


def callout(title: str, text: str) -> Table:
    content = [[[p(title, "h2"), p(text)]]]
    result = Table(content, colWidths=[WIDTH], hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 13),
        ("RIGHTPADDING", (0, 0), (-1, -1), 13),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return result


def scaled_image(path: Path, max_height: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(WIDTH / width, max_height / height)
    result = Image(str(path), width=width * scale, height=height * scale)
    result.hAlign = "CENTER"
    return result


def decorate(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(46, PAGE_HEIGHT - 43, PAGE_WIDTH - 46, PAGE_HEIGHT - 43)
    canvas.setFont("Helvetica-Bold", 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(46, PAGE_HEIGHT - 34, "FREIGHT RATE PREDICTION  /  TECHNICAL REPORT")
    canvas.line(46, 43, PAGE_WIDTH - 46, 43)
    canvas.setFont("Helvetica", 7.3)
    canvas.drawString(46, 30, "Machine Learning Engineer Assessment | Candidate solution")
    canvas.drawRightString(PAGE_WIDTH - 46, 30, f"{doc.page}")
    canvas.restoreState()


def require_content(content: dict) -> None:
    keys = ["summary", "data", "features", "evaluation", "december", "reproducibility"]
    missing = [key for key in keys if key not in content]
    if missing:
        raise ValueError(f"Report metadata is missing required sections: {', '.join(missing)}")
    for key in keys:
        if not isinstance(content[key], dict):
            raise ValueError(f"Report metadata section {key!r} must be an object")


def make_report_data(metrics: dict, audit: dict) -> dict:
    """Translate recorded metrics/audit into reader-facing report content.

    The dates below are the prespecified assessment development protocol. Row
    counts are recomputed from the audit, never estimated from percentages.
    """
    train = audit["tables"]["train"]
    validation = audit["tables"]["validation"]
    chart = audit["tables"]["december_chart"]
    generalization = audit["cross_table_checks"]["validation_generalization"]
    monthly = train["dates"]["monthly_rows"]
    jan_jun = sum(count for month, count in monthly.items() if month <= "2025-06")
    jul_aug = sum(count for month, count in monthly.items() if "2025-07" <= month <= "2025-08")
    sep_oct = sum(count for month, count in monthly.items() if month >= "2025-09")
    selected = metrics["selected_model"]
    holdout = metrics["holdout_results"]
    full = holdout["full"]
    reduced = holdout["reduced"]
    baseline = holdout["baseline"]
    selected_name = selected["name"]
    december = metrics["december"]

    def number(value: float, digits: int = 2) -> str:
        return f"{float(value):,.{digits}f}"

    def pct(value: float) -> str:
        return f"{100 * float(value):.2f}%"

    def duplicates(table: dict) -> int:
        value = table["duplicate_load_ids"]
        return int(value["extra_rows"] if isinstance(value, dict) else value)

    def score_row(name: str, stage: str, result: dict) -> list[str]:
        return [name, stage, number(result["rmse"]), number(result["mae"]), pct(result["wape"])]

    candidates = metrics["candidate_results"]
    if isinstance(candidates, dict):
        candidates = [{"name": name, **result} for name, result in candidates.items()]
    selection_best = next(result for result in candidates if result["name"] == selected_name)
    reduced_name = december["model"]
    selection_reduced = next(result for result in candidates if result["name"] == reduced_name)
    comparison = [
        score_row("Primary selected model", "Jul-Aug selection", selection_best),
        score_row("Reduced-input model", "Jul-Aug selection", selection_reduced),
        score_row("Distance/equipment baseline", "Sep-Oct holdout", baseline),
        score_row("Primary selected model", "Sep-Oct holdout", full),
        score_row("Reduced-input model", "Sep-Oct holdout", reduced),
    ]
    raw_target = selected["target"]
    target_descriptions = {
        "raw": "The model fits posted_rate directly with RMSE loss, retaining the influence of expensive loads.",
        "log": "The model fits log(posted_rate) and exponentiates predictions. This reduces the influence of rare expensive loads; selection still uses RMSE in original dollar units.",
        "log_rpm": "The model fits log(posted_rate / distance), then exponentiates and multiplies by the supplied distance. This keeps the rate-per-mile structure while selection uses RMSE in dollars.",
    }
    if raw_target not in target_descriptions:
        raise ValueError(f"Document the selected target transform: {raw_target!r}")
    improvement = 1 - float(full["rmse"]) / float(baseline["rmse"])
    diagnostic_rows = []
    subgroup = metrics.get("subgroup_metrics", {})
    monthly_metrics = subgroup.get("month", [])
    if isinstance(monthly_metrics, dict):
        monthly_metrics = [{"group": label, **result} for label, result in monthly_metrics.items()]
    for result in monthly_metrics:
        diagnostic_rows.append([result["group"], f"{result['rows']:,}", number(result["rmse"]), pct(result["wape"])])
    for kind in ["equipment", "weight_quality"]:
        for result in subgroup.get(kind, []):
            label = result["group"].replace("missing_or_invalid", "Missing / invalid weight").replace("valid", "Valid weight")
            diagnostic_rows.append([label, f"{result['rows']:,}", number(result["rmse"]), pct(result["wape"])])
    stress = metrics.get("cold_city_stress")
    stress_notes = []
    if stress:
        stress_metrics = stress["metrics"]
        stress_notes = [
            f"Controlled cold-city stress: deterministically select {len(stress['heldout_cities_in_hash_order'])} January-June cities without labels, remove all training loads touching them, and refit the already selected primary configuration on {stress['training']['retained_rows']:,} retained rows.",
            f"Evaluate {stress_metrics['rows']:,} July-August loads touching those cities: RMSE ${number(stress_metrics['rmse'])}, MAE ${number(stress_metrics['mae'])}, WAPE {pct(stress_metrics['wape'])}. This post-selection test does not tune the model and does not measure the actual November-December unseen cities.",
        ]

    return {
        "summary": {
            "subtitle": "Machine Learning Engineer Assessment\nA documented workflow from raw inputs to reproducible predictions.",
            "cards": [
                {"value": "$" + number(full["rmse"]), "label": "Primary model holdout RMSE"},
                {"value": "$" + number(full["mae"]), "label": "Primary model holdout MAE"},
                {"value": pct(full["wape"]), "label": "Primary model holdout WAPE"},
            ],
            "decision": [
                f"Use {selected_name} for the 12,000 November-December loads, then use {reduced_name} for the fixed December scenario whose input schema has fewer fields.",
                "Choose model configurations on a July-August temporal selection window, evaluate the frozen choice on September-October, and refit the two selected configurations on all January-October labeled rows for submission.",
            ],
            "findings": [
                "The dataset calls for future-period validation: the labeled development data ends in October and the final labels are withheld for November-December.",
                "Weights contain missing and nonpositive values, new cities appear in validation, and the quote signal changes its relationship with price between development months. These are explicit pipeline and feature-policy decisions.",
                f"The primary model lowers holdout RMSE by {100 * improvement:.1f}% versus the distance/equipment baseline. This is an internal retrospective result, not the organizer's unseen validation score.",
            ],
            "caveat": "The supplied scorer checks IDs, schema, finite positive predictions and the December grid. It does not calculate accuracy. November-December prediction accuracy remains unknown until the organizer scores the withheld labels.",
        },
        "data": {
            "introduction": "Audit all supplied files without modifying them. Preserve the identifier and input schemas, and keep source hashes so the results can be tied to the exact inputs.",
            "scope_headers": ["Input", "Rows", "Dates", "Role"],
            "scope_rows": [
                ["train_test.csv", f"{train['rows']:,}", f"{train['dates']['min']} to {train['dates']['max']}", "Labeled development data"],
                ["validation.csv", f"{validation['rows']:,}", f"{validation['dates']['min']} to {validation['dates']['max']}", "Final prediction only; no labels"],
                ["December chart inputs", str(chart["rows"]), "2025-12-01 to 2025-12-31", "Fixed lane scenario; fewer features"],
            ],
            "quality_headers": ["Issue", "Train / validation", "Treatment or implication"],
            "quality_rows": [
                ["Missing weight", f"{train['missing']['weight']} / {validation['missing']['weight']}", "Keep a missing/invalid flag; impute numeric values using training-only medians."],
                ["Nonpositive weight", f"{train['numeric']['weight']['negative'] + train['numeric']['weight']['zero']} / {validation['numeric']['weight']['negative'] + validation['numeric']['weight']['zero']}", "Treat as missing. Do not take absolute values or discard the full load."],
                ["Missing market_index", f"{train['missing']['market_index']} / {validation['missing']['market_index']}", "Training-only numeric imputation plus a missingness indicator in the primary model."],
                ["Duplicate identifiers / rows", f"{duplicates(train)} / {duplicates(validation)} extra ID rows; {train['exact_duplicate_rows']} / {validation['exact_duplicate_rows']} exact duplicate rows", "Keep every supplied load; preserve the template's order when writing predictions."],
                ["Unseen locations and lanes", f"{len(generalization['unseen_cities'])} new cities; {generalization['rows_with_unseen_city']:,} affected rows", f"{generalization['rows_with_unseen_ordered_lane']:,} validation loads also have an unseen directed lane. Support unseen categorical values; the primary model retains supplied geography."],
                ["Rare expensive targets", f"Maximum ${number(train['numeric']['posted_rate']['summary']['max'])}", "Retain labeled rates: size alone does not prove an error. Compare losses/target transforms and inspect error concentration."],
            ],
            "notes": [
                "The audit finds no numeric parse failures, nonfinite numeric inputs, invalid ISO dates, overlapping load IDs or exact duplicate rows. Coordinate checks verify mathematical bounds only; geographic correctness is unverified.",
                "Counts above describe source inputs. Data-quality transformations are fitted independently inside each training partition, then reused unchanged on its later evaluation partition.",
            ],
        },
        "features": {
            "introduction": "One deterministic feature function is shared by training and inference. load_id is used only to align outputs; posted_rate is the target and never enters the feature matrix.",
            "headers": ["Group", "Derived or retained features", "Availability"],
            "rows": [
                ["Shipment / lane", "Distance, log/square-root distance, cleaned weight, weight missingness; pickup, delivery, equipment and directed lane", "Primary and reduced models"],
                ["Calendar", "Day of week/month/year, days from a fixed origin, weekend flag; weekly, annual and monthly sine/cosine terms", "Derived only from supplied date"],
                ["Geography", "Supplied pickup/delivery coordinates and coordinate differences", "Primary model only"],
                ["Market", "market_index and its missingness flag", "Primary model only"],
                ["Quote signal", "Excluded from both submission models; quote-based candidates remain diagnostic benchmarks", "Provenance and stability unresolved"],
                ["Missing values", "Categorical missing token; training-only numeric medians; no future-data lookup table", "Fitted on training partition only"],
            ],
            "rationale": [
                "CatBoost learns nonlinear shipment, calendar and location interactions and supports unseen categories. Ridge provides a regularized comparator with one-hot categories and distance interactions.",
                target_descriptions[raw_target],
                f"Fixed configuration: {selected['params']['iterations']} iterations, depth {selected['params']['depth']}, learning rate {selected['params']['learning_rate']}, seed {selected['params']['random_seed']}. Select by dollar RMSE; the later holdout does not tune features, loss or stopping.",
            ],
            "availability": [
                "January-August quote_signal associations reverse sign and become nearly zero in August. Exclude this unstable field conservatively; the observations do not prove target leakage. Observed rate per mile remains a training target/diagnostic only.",
            ],
            "caveat": "Signal definitions and timestamps are unspecified. The primary model assumes market_index is available before pricing. Verify its calculation and timing before operational use.",
        },
        "evaluation": {
            "introduction": "Preserve time order to match the organizer's future-period task. Keep September-October separate from candidate comparison, then evaluate the frozen selections once on that later window.",
            "split_headers": ["Stage", "Training data", "Later data / use"],
            "split_rows": [
                ["Candidate selection", f"Jan-Jun 2025: {jan_jun:,} rows", f"Jul-Aug 2025: {jul_aug:,} rows; select models"],
                ["Locked holdout", f"Jan-Aug 2025: {jan_jun + jul_aug:,} rows", f"Sep-Oct 2025: {sep_oct:,} rows; report accuracy"],
                ["Final refit", f"Jan-Oct 2025: {train['rows']:,} rows", "Nov-Dec: 12,000 unlabelled rows; no tuning"],
            ],
            "result_headers": ["Model", "Evaluation window", "RMSE ($)", "MAE ($)", "WAPE"],
            "result_widths": [.28, .25, .16, .16, .15],
            "result_rows": comparison,
            "notes": [
                f"Primary configuration: {selected_name}. Reduced configuration: {reduced_name}. All candidate results and configurations are retained in artifacts/metrics.json and experiments/results.json.",
                "RMSE = sqrt(mean squared dollar error); MAE = mean absolute dollar error; WAPE = sum absolute error / sum observed rates. Selection and holdout results use different training histories and are not interchangeable.",
            ],
            "interpretation": [
                f"Primary holdout R-squared is {number(full['r2'], 4)}. This summarizes the supplied development period; it cannot establish performance on new cities or market conditions in November-December.",
            ],
            "stress_notes": stress_notes,
        },
        "diagnostics": {
            "introduction": "Every September-October label is retained. Inspect month, equipment and weight-quality slices rather than relying on the aggregate score alone.",
            "headers": ["Holdout group", "Rows", "RMSE ($)", "WAPE"],
            "rows": diagnostic_rows,
            "stress_notes": stress_notes,
            "chart_path": "reports/holdout_diagnostics.png",
            "caption": "All-label holdout diagnostics: the observed/predicted panel uses log axes for visibility; the error histogram stays in original dollar units. Rare expensive loads account for substantial dollar-error tails.",
        },
        "december": {
            "introduction": "Predict 31 daily loads from Lexington to Fort Wayne, keeping distance at 360 miles, equipment as Dry Van and weight at 32,000 lb. Only the supplied date varies.",
            "chart_path": "reports/scorer_results/candidate_december.png",
            "caption": "Figure: candidate_december.png generated by the organizer's unmodified score.py from the completed December CSV.",
            "method": [
                f"The scenario has no coordinates, market_index or quote_signal. Use the separately selected {reduced_name}, which is trained and evaluated using the same six base input columns, rather than supplying invented future signals.",
                "The reduced model shares shipment, lane and calendar transformations with the primary model. It needs no external city lookup, forecast market index or future quote. Refit it on all January-October labeled rows before producing December predictions.",
                f"The reduced model's September-October RMSE is ${number(reduced['rmse'])}. That result measures its complete reduced feature contract across the held-out loads, not accuracy for this particular December lane.",
            ],
            "caveat": "This is a model scenario, not an observed December price curve. Daily shape reflects learned calendar effects under fixed inputs. One year of January-October history cannot establish a repeated December seasonal pattern, and market changes remain unobserved here.",
        },
        "reproducibility": {
            "introduction": "The repository contains the feature/model code, fixed development protocol, recorded metrics, final predictions and the report source. Follow README.md to reproduce the submission; the candidate records and submits their own Loom walkthrough.",
            "artifact_headers": ["Artifact", "Purpose"],
            "artifact_rows": [
                ["validation_predictions.csv", "Exactly load_id,predicted_rate; 12,000 loads aligned to the supplied template"],
                ["submission/december_chart_predictions.csv", "31 completed daily rows; original seven-column schema preserved"],
                ["artifacts/full_model.joblib / reduced_model.joblib", "Final refitted models and training-only preprocessing state"],
                ["artifacts/metrics.json", "Candidate selection, holdout scores, configuration and reproduction metadata"],
                ["reports/data_audit.json", "Input hashes, quality counts and distribution checks"],
                ["reports/freight_rate_report.pdf", "This report, including the complete official December chart"],
            ],
            "commands": [
                "python -m pip install -r requirements.txt",
                "python -m src.run_all",
                "python -m src.train --recompute-selection",
            ],
            "verification": [
                "The default complete workflow audits inputs, trains/refits selected configurations, writes both prediction files, runs the original scorer and regenerates this PDF. Recompute selection to independently rerun the recorded candidate comparison.",
                "Official scorer acceptance establishes file structure and finite positive predictions. Meaningful pipeline tests additionally cover input sanitization, target/ID exclusion, missing values and the reduced-input schema.",
            ],
            "limitations": [
                "Future labels are unavailable. Temporal holdout is a defensible retrospective estimate, but final period performance is still unknown.",
                "All holdout cities are represented in its training history. A separate controlled cold-city stress provides supporting evidence, without establishing accuracy on the actual new validation cities.",
                "Rare expensive loads can dominate dollar RMSE. Preserve them, inspect subgroup errors and validate unusual source prices before deciding on any future removal or correction policy.",
                "Before operational use, verify signal timestamps, collect additional seasonal cycles and backtest on several rolling origin windows, including truly unseen locations.",
            ],
            "provenance": "Evidence: supplied assessment PDF/README and unchanged scorer; recorded read-only data audit; recorded July-August candidate results and September-October holdout results. Input/model hashes, package versions, random seed and thread count are recorded in artifacts/metrics.json.",
        },
    }


def build_report(content: dict, output: Path, root: Path = ROOT) -> None:
    require_content(content)
    summary = content["summary"]
    data = content["data"]
    features = content["features"]
    evaluation = content["evaluation"]
    december = content["december"]
    reproducibility = content["reproducibility"]
    story: list = []

    # Page 1: decision and measurable outcome.
    story.extend([
        p("01 / APPROACH", "eyebrow"),
        p("Predicting freight rates\nfor future loads", "title"),
        p(summary["subtitle"], "subtitle"),
        cards(summary["cards"]),
        Spacer(1, 16),
        p("Decision", "h2"),
    ])
    story.extend(paragraphs(summary["decision"]))
    story.append(p("What this submission demonstrates", "h2"))
    story.extend(paragraphs(summary["findings"]))
    if summary.get("caveat"):
        story.append(callout("Evidence boundary", summary["caveat"]))
    story.append(PageBreak())

    # Page 2: source scope and data quality.
    story.extend(section("02 / DATA", "Data scope and preparation", data["introduction"]))
    story.append(data_table(data["scope_headers"], data["scope_rows"], [WIDTH * .29, WIDTH * .13, WIDTH * .28, WIDTH * .30]))
    story.append(p("Data-quality checks and treatment", "h2"))
    story.append(data_table(data["quality_headers"], data["quality_rows"], [WIDTH * .24, WIDTH * .22, WIDTH * .54]))
    story.append(Spacer(1, 9))
    story.extend(paragraphs(data["notes"], "small"))
    story.append(PageBreak())

    # Page 3: feature contract and production assumptions.
    story.extend(section("03 / FEATURES", "A reproducible feature pipeline", features["introduction"]))
    story.append(data_table(features["headers"], features["rows"], [WIDTH * .22, WIDTH * .47, WIDTH * .31]))
    story.append(p("Why this model", "h2"))
    story.extend(paragraphs(features["rationale"]))
    story.append(p("Feature availability and leakage", "h2"))
    story.extend(paragraphs(features["availability"]))
    if features.get("caveat"):
        story.append(callout("Operational assumption", features["caveat"]))
    story.append(PageBreak())

    # Page 4: temporal splits, recorded candidate scores and diagnostics.
    story.extend(section("04 / VALIDATION", "Chronological validation", evaluation["introduction"]))
    story.append(data_table(evaluation["split_headers"], evaluation["split_rows"], [WIDTH * .24, WIDTH * .36, WIDTH * .40]))
    story.append(p("Model comparison", "h2"))
    results_headers = evaluation["result_headers"]
    widths = evaluation.get("result_widths")
    if widths:
        widths = [WIDTH * fraction for fraction in widths]
    story.append(data_table(results_headers, evaluation["result_rows"], widths))
    story.append(Spacer(1, 9))
    story.extend(paragraphs(evaluation["notes"], "small"))
    if evaluation.get("diagnostic_rows"):
        story.append(p("Held-out diagnostics", "h2"))
        diagnostic_headers = evaluation["diagnostic_headers"]
        story.append(data_table(diagnostic_headers, evaluation["diagnostic_rows"], [WIDTH / len(diagnostic_headers)] * len(diagnostic_headers)))
    if evaluation.get("interpretation"):
        story.extend(paragraphs(evaluation["interpretation"]))
    if evaluation.get("stress_notes"):
        story.append(p("Controlled cold-city stress", "h2"))
        story.extend(paragraphs(evaluation["stress_notes"], "small"))
    story.append(PageBreak())

    # Page 5: chart produced by the official, unmodified score.py.
    story.extend(section("05 / DECEMBER", "A fixed lane, 31 future dates", december["introduction"]))
    chart = root / december["chart_path"]
    if not chart.is_file():
        raise FileNotFoundError(f"Official scorer chart is required: {chart}")
    story.append(scaled_image(chart, 3.45 * inch))
    story.append(Spacer(1, 9))
    story.append(p(december["caption"], "small"))
    story.append(p("Handling the reduced input contract", "h2"))
    story.extend(paragraphs(december["method"]))
    story.append(callout("Interpretation", december["caveat"]))
    story.append(PageBreak())

    # Page 6: handoff, limitations, provenance and acceptance checks.
    story.extend(section("06 / REPRODUCIBILITY", "Run, inspect and submit", reproducibility["introduction"]))
    story.append(data_table(reproducibility["artifact_headers"], reproducibility["artifact_rows"], [WIDTH * .43, WIDTH * .57]))
    story.append(p("Reproduction", "h2"))
    story.extend(paragraphs(reproducibility["commands"], "code"))
    story.extend(paragraphs(reproducibility["verification"], "small"))
    story.append(p("Limitations and follow-up", "h2"))
    story.extend(paragraphs(reproducibility["limitations"]))
    if reproducibility.get("provenance"):
        story.append(p(reproducibility["provenance"], "small"))

    if content.get("include_diagnostics_page"):
        diagnostics = content["diagnostics"]
        story.append(PageBreak())
        story.extend(section("07 / DIAGNOSTICS", "Where the errors remain", diagnostics["introduction"]))
        story.append(data_table(diagnostics["headers"], diagnostics["rows"], [WIDTH * .40, WIDTH * .17, WIDTH * .22, WIDTH * .21]))
        if diagnostics.get("stress_notes"):
            story.append(p("Controlled generalization stress", "h2"))
            story.extend(paragraphs(diagnostics["stress_notes"], "small"))
        story.append(Spacer(1, 5))
        diagnostic_chart = root / diagnostics["chart_path"]
        if diagnostic_chart.is_file():
            story.append(scaled_image(diagnostic_chart, 2.45 * inch))
            story.append(p(diagnostics["caption"], "small"))

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=46, leftMargin=46,
        topMargin=60, bottomMargin=58, title="Freight Rate Prediction Assessment",
        author="Candidate solution", subject="Data quality, temporal validation, predictions and reproducibility",
    )
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)


def render_preview(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Render all pages headlessly for visual quality assurance."""
    import pymupdf

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(pdf_path) as document:
        for number, page in enumerate(document, start=1):
            path = output_dir / f"page_{number:02d}.png"
            page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(path)
            paths.append(path)
    return paths


def inspect_pdf(pdf_path: Path) -> dict:
    """Check rendered text bounds and required sections; visual QA is still needed."""
    import pymupdf

    pages = []
    clipped = []
    with pymupdf.open(pdf_path) as document:
        all_text = "\n".join(page.get_text() for page in document)
        for number, page in enumerate(document, start=1):
            text = page.get_text()
            pages.append({"page": number, "text_characters": len(text), "embedded_images": len(page.get_images())})
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                x0, y0, x1, y1 = block["bbox"]
                if x0 < -0.1 or y0 < -0.1 or x1 > page.rect.width + 0.1 or y1 > page.rect.height + 0.1:
                    clipped.append({"page": number, "bbox": [x0, y0, x1, y1]})
        required = ["01 / APPROACH", "02 / DATA", "03 / FEATURES", "04 / VALIDATION", "05 / DECEMBER", "06 / REPRODUCIBILITY"]
        missing = [title for title in required if title not in all_text]
        return {
            "page_count": len(document), "pages": pages,
            "text_blocks_outside_page": clipped,
            "missing_required_sections": missing,
            "automated_checks_passed": len(document) in range(5, 8) and not clipped and not missing,
            "scope": "Selectable-text and page-bound checks only; inspect rendered page images for visual layout.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-data", type=Path, default=ROOT / "artifacts/report_data.json")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts/metrics.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/data_audit.json")
    parser.add_argument("--cold-city-stress", type=Path, default=ROOT / "reports/cold_city_stress.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/freight_rate_report.pdf")
    parser.add_argument("--render-dir", type=Path, help="Optional directory for headless page PNG previews")
    parser.add_argument("--qa-json", type=Path, help="Optional machine-readable PDF structural QA")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if args.cold_city_stress.is_file():
        metrics["cold_city_stress"] = json.loads(args.cold_city_stress.read_text(encoding="utf-8"))
    content = make_report_data(
        metrics,
        json.loads(args.audit.read_text(encoding="utf-8")),
    )
    args.report_data.parent.mkdir(parents=True, exist_ok=True)
    args.report_data.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    build_report(content, args.output)
    print(f"Created report: {args.output}")
    if args.render_dir:
        paths = render_preview(args.output, args.render_dir)
        print(f"Rendered {len(paths)} pages to {args.render_dir}")
    if args.qa_json:
        inspection = inspect_pdf(args.output)
        args.qa_json.parent.mkdir(parents=True, exist_ok=True)
        args.qa_json.write_text(json.dumps(inspection, indent=2) + "\n", encoding="utf-8")
        print(f"PDF structural checks: {inspection['automated_checks_passed']}")


if __name__ == "__main__":
    main()
