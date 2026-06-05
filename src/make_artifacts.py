from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    ListFlowable,
    ListItem,
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "demo"
REPORT = ROOT / "reports" / "technical_report.pdf"
SLIDES = ROOT / "presentation" / "recommendation_presentation.pdf"


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=14))
    styles.add(ParagraphStyle(name="Heading", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.2, leading=10))
    styles.add(ParagraphStyle(name="BodyJust", parent=styles["BodyText"], fontSize=9.5, leading=12, alignment=TA_LEFT))
    return styles


def _table(data, col_widths=None, font_size=8):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
    ]))
    return t


def build_report():
    styles = _styles()
    story = []
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(OUT / "metrics.csv")
    eda = json.loads((OUT / "eda_summary.json").read_text())

    doc = SimpleDocTemplate(str(REPORT), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)

    story.append(Paragraph("Technical Report: Personalized Netflix Recommendation System", styles["TitleCenter"]))
    story.append(Paragraph("Problem Understanding", styles["Heading"]))
    story.append(Paragraph(
        "The objective is to learn user preferences from historical user-movie ratings, predict unseen ratings, and generate useful Top-K content recommendations. The solution must balance rating-prediction accuracy with ranking quality, because a streaming platform ultimately needs relevant movies near the top of the recommendation list.",
        styles["BodyJust"],
    ))

    story.append(Paragraph("Dataset and EDA", styles["Heading"]))
    eda_table = [["Metric", "Value"],
                 ["Ratings", f"{eda['n_ratings']:,}"],
                 ["Users", f"{eda['n_users']:,}"],
                 ["Movies", f"{eda['n_movies']:,}"],
                 ["Mean rating", f"{eda['rating_mean']:.3f}"],
                 ["Sparsity", f"{eda['sparsity']:.2%}"],
                 ["Date range", f"{eda['date_min']} to {eda['date_max']}"]]
    story.append(_table(eda_table, col_widths=[2.0*inch, 3.6*inch]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The Netflix-style recommendation problem is highly sparse: users rate only a small subset of movies. This makes naive popularity recommendations insufficient and motivates collaborative filtering and latent-factor approaches.",
        styles["BodyJust"],
    ))
    for fig in ["rating_distribution.png", "top_movies.png"]:
        path = OUT / "figures" / fig
        if path.exists():
            story.append(Spacer(1, 8))
            story.append(Image(str(path), width=5.4*inch, height=3.0*inch))

    story.append(PageBreak())
    story.append(Paragraph("Methodology", styles["Heading"]))
    methods = [
        "Bias baseline: global mean + user bias + movie bias. This is the sanity-check model.",
        "Item-based collaborative filtering: computes movie-to-movie similarity from rating patterns and recommends movies similar to those the user has rated highly.",
        "Matrix factorization: learns latent user and movie vectors using stochastic gradient descent with bias terms.",
        "Temporal train-test split: older ratings are used for training and newer ratings are held out for testing, avoiding random time leakage.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(m, styles["BodyJust"])) for m in methods], bulletType="bullet"))

    story.append(Paragraph("Evaluation Strategy", styles["Heading"]))
    story.append(Paragraph(
        "RMSE and MAE evaluate predicted rating accuracy. MAP@10 evaluates whether relevant held-out movies appear near the top of each user's recommendation list. A movie is treated as relevant when its actual rating is greater than or equal to 3.5.",
        styles["BodyJust"],
    ))

    story.append(Paragraph("Experimental Results", styles["Heading"]))
    metric_display = metrics.copy()
    metric_display = metric_display.rename(columns={"Precision@10": "P@10", "Recall@10": "R@10", "HitRate@10": "Hit@10", "EvaluatedUsers": "Users"})
    for col in ["RMSE", "MAE", "MAP@10", "P@10", "R@10", "Hit@10"]:
        metric_display[col] = metric_display[col].map(lambda x: f"{x:.4f}")
    table_data = [metric_display.columns.tolist()] + metric_display.astype(str).values.tolist()
    story.append(_table(table_data, col_widths=[1.65*inch, .70*inch, .65*inch, .75*inch, .70*inch, .70*inch, .75*inch, .50*inch], font_size=6.8))
    story.append(Spacer(1, 8))
    best = metrics.iloc[0]
    story.append(Paragraph(
        f"On the bundled demo dataset, the strongest ranking model is {best['model']} with MAP@10={best['MAP@10']:.4f}. For the full Netflix dataset, these numbers should be regenerated after increasing the training subset size.",
        styles["BodyJust"],
    ))

    story.append(Paragraph("Recommendation Generation and Explainability", styles["Heading"]))
    rec_files = sorted((OUT / "recommendations").glob("*_sample_recommendations.csv"))
    if rec_files:
        recs = pd.read_csv(rec_files[0]).head(10)
        recs = recs[["user_id", "rank", "movie_id", "title", "score"]]
        recs["score"] = recs["score"].map(lambda x: f"{x:.3f}")
        story.append(_table([recs.columns.tolist()] + recs.astype(str).values.tolist(), col_widths=[.6*inch, .45*inch, .55*inch, 3.4*inch, .65*inch], font_size=7.2))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The item-based model supports explanations by identifying previously rated movies that are most similar to a recommended movie. This improves transparency and makes the dashboard more useful for judging recommendation quality.",
        styles["BodyJust"],
    ))

    story.append(PageBreak())
    story.append(Paragraph("Key Insights", styles["Heading"]))
    insights = [
        "Sparse interactions mean that filtering very inactive users/items improves small-scale experiments.",
        "A bias baseline is hard to beat on RMSE, so ranking metrics are necessary to judge actual recommendation usefulness.",
        "Item-based collaborative filtering gives strong explainability and is practical for a dashboard prototype.",
        "Matrix factorization is scalable in concept but needs careful tuning and larger data to show its full advantage.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(x, styles["BodyJust"])) for x in insights], bulletType="bullet"))
    story.append(Paragraph("Failure Cases and Limitations", styles["Heading"]))
    limitations = [
        "Cold-start users and movies cannot be personalized without additional metadata or onboarding signals.",
        "Popularity bias can dominate recommendations when user history is short.",
        "The demo metrics are not competition metrics; final numbers must be generated on the official Netflix Prize data or a clearly stated subset.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(x, styles["BodyJust"])) for x in limitations], bulletType="bullet"))
    story.append(Paragraph("Future Improvements", styles["Heading"]))
    future = [
        "Add ensemble blending of baseline, item-CF, and matrix-factorization scores.",
        "Add diversity re-ranking so the top list is not too narrow.",
        "Use movie metadata for cold-start and explainable hybrid recommendations.",
        "Scale evaluation with sampled candidate sets and batch scoring for larger Netflix subsets.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(x, styles["BodyJust"])) for x in future], bulletType="bullet"))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Conclusion", styles["Heading"]))
    story.append(Paragraph(
        "The project delivers a complete recommender-system pipeline with EDA, model comparison, RMSE, MAP@10, Top-K recommendations, explainability, and a dashboard. The design is simple enough to reproduce but strong enough to defend technically.",
        styles["BodyJust"],
    ))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0]-42, 24, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def slide_header(c, title):
    w, h = landscape(A4)
    c.setFillColor(colors.HexColor("#111827"))
    c.rect(0, h-0.55*inch, w, 0.55*inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.45*inch, h-0.38*inch, title)
    c.setFillColor(colors.black)


def bullet(c, x, y, text, size=15):
    c.setFont("Helvetica", size)
    c.drawString(x, y, u"•")
    text_obj = c.beginText(x+0.22*inch, y)
    text_obj.setFont("Helvetica", size)
    max_chars = 82
    words = text.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > max_chars:
            text_obj.textLine(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        text_obj.textLine(line)
    c.drawText(text_obj)


def build_slides():
    SLIDES.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(OUT / "metrics.csv")
    eda = json.loads((OUT / "eda_summary.json").read_text())
    w, h = landscape(A4)
    c = canvas.Canvas(str(SLIDES), pagesize=landscape(A4))

    # 1
    c.setFillColor(colors.HexColor("#111827")); c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(w/2, h*0.62, "Personalized Netflix Recommendation System")
    c.setFont("Helvetica", 16)
    c.drawCentredString(w/2, h*0.52, "EDA + Collaborative Filtering + Matrix Factorization + MAP@10 Evaluation")
    c.setFont("Helvetica", 12)
    c.drawCentredString(w/2, h*0.43, "Complete reproducible project with dashboard, report, and recommendation examples")
    c.showPage()

    # 2
    slide_header(c, "Problem Overview")
    ys = [h-1.2*inch, h-1.75*inch, h-2.3*inch, h-2.85*inch]
    texts = [
        "Learn user preferences from historical user-movie ratings.",
        "Predict unseen ratings and generate personalized Top-K recommendations.",
        "Identify similar content and make recommendations explainable.",
        "Evaluate both rating accuracy and ranking quality.",
    ]
    for y, t in zip(ys, texts): bullet(c, 0.7*inch, y, t)
    c.showPage()

    # 3
    slide_header(c, "EDA Findings")
    stats = [
        f"Demo ratings: {eda['n_ratings']:,}; users: {eda['n_users']:,}; movies: {eda['n_movies']:,}.",
        f"Mean rating: {eda['rating_mean']:.2f}; sparsity: {eda['sparsity']:.2%}.",
        "Ratings are sparse, so pure popularity is not enough for personalization.",
    ]
    for i, t in enumerate(stats): bullet(c, 0.7*inch, h-(1.2+0.55*i)*inch, t)
    fig = OUT / "figures" / "rating_distribution.png"
    if fig.exists(): c.drawImage(str(fig), 4.6*inch, 1.1*inch, width=4.4*inch, height=3.0*inch, preserveAspectRatio=True)
    c.showPage()

    # 4
    slide_header(c, "Model Design")
    model_bullets = [
        "Bias baseline: global mean + user/movie bias for a hard sanity-check benchmark.",
        "Item-based collaborative filtering: recommends movies similar to those the user liked.",
        "Matrix factorization: learns latent user/movie vectors for hidden preference patterns.",
        "Temporal split prevents future information from leaking into training.",
    ]
    for i, t in enumerate(model_bullets): bullet(c, 0.7*inch, h-(1.2+0.6*i)*inch, t)
    c.showPage()

    # 5
    slide_header(c, "Evaluation Metrics")
    eval_bullets = [
        "RMSE: measures predicted rating error.",
        "MAE: interpretable average absolute rating error.",
        "MAP@10: measures ranking quality for the Top-10 list.",
        "Relevant movie definition: actual rating >= 3.5.",
    ]
    for i, t in enumerate(eval_bullets): bullet(c, 0.7*inch, h-(1.2+0.6*i)*inch, t)
    c.showPage()

    # 6
    slide_header(c, "Experimental Results")
    c.setFont("Helvetica-Bold", 11)
    x0, y0 = 0.6*inch, h-1.25*inch
    cols = ["model", "RMSE", "MAE", "MAP@10", "HitRate@10"]
    widths = [2.5*inch, .9*inch, .9*inch, 1.0*inch, 1.1*inch]
    x = x0
    for col, cw in zip(cols, widths):
        c.drawString(x, y0, col); x += cw
    c.setFont("Helvetica", 10)
    for r, (_, row) in enumerate(metrics.iterrows(), start=1):
        y = y0 - r*0.35*inch
        x = x0
        vals = [row["model"], f"{row['RMSE']:.3f}", f"{row['MAE']:.3f}", f"{row['MAP@10']:.3f}", f"{row['HitRate@10']:.3f}"]
        for val, cw in zip(vals, widths):
            c.drawString(x, y, str(val)[:32]); x += cw
    best = metrics.iloc[0]
    bullet(c, 0.7*inch, 1.7*inch, f"Best demo ranking model: {best['model']} with MAP@10={best['MAP@10']:.3f}.", size=14)
    c.showPage()

    # 7
    slide_header(c, "Recommendation Example")
    rec_files = sorted((OUT / "recommendations").glob("*_sample_recommendations.csv"))
    if rec_files:
        recs = pd.read_csv(rec_files[0]).head(5)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.7*inch, h-1.15*inch, "Sample Top-5 recommendations")
        c.setFont("Helvetica", 11)
        for i, row in recs.iterrows():
            c.drawString(0.9*inch, h-(1.55+0.38*i)*inch, f"{int(row['rank'])}. {row['title']} - score {row['score']:.2f}")
    bullet(c, 0.7*inch, 1.5*inch, "The dashboard lets users inspect recommendation scores and explanations.", size=14)
    c.showPage()

    # 8
    slide_header(c, "Key Insights and Next Steps")
    last_bullets = [
        "MAP@10 is the metric that makes this a recommender, not just a rating regressor.",
        "Item-CF is strong for explainability; MF is stronger after larger-scale tuning.",
        "Next: train larger subsets, blend models, add diversity re-ranking, and implement cold-start logic.",
        "The project is GitHub-ready: code, docs, report, slides, dashboard, and reproducible commands are included.",
    ]
    for i, t in enumerate(last_bullets): bullet(c, 0.7*inch, h-(1.2+0.62*i)*inch, t)
    c.save()


if __name__ == "__main__":
    build_report()
    build_slides()
    print(f"Created {REPORT}")
    print(f"Created {SLIDES}")
