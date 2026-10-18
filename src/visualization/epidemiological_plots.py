"""Epidemiological Visualization Module for EpiWatch Phase 2."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("epidemiological_plots")

# Apply clean plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def plot_epidemiological_visualizations(
    df_epi: pd.DataFrame,
    ts_moving_avg: pd.DataFrame,
    inc_df: pd.DataFrame,
    cfr_df: pd.DataFrame,
    hosp_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    season_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    anom_df: pd.DataFrame,
    output_dir: Path
):
    """Generate all required epidemiological plots and save to PNG.

    Figures generated in output_dir (reports/figures/):
    - disease_case_trends.png
    - {disease}_weekly_cases.png
    - {disease}_moving_average.png
    - {disease}_monthly_cases.png
    - {disease}_yearly_cases.png
    - moving_averages.png
    - monthly_trends.png
    - yearly_trends.png
    - disease_seasonality.png
    - growth_rates.png
    - incidence_trends.png
    - positivity_over_time.png
    - cases_vs_hospitalizations.png
    - hospitalizations_vs_deaths.png
    - potential_anomalies.png
    """
    logger.info("Generating epidemiological visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    diseases = df["disease"].unique()

    # 1. Overall Disease Case Trends
    plt.figure(figsize=(12, 6))
    for d in diseases:
        df_d = df[df["disease"] == d].set_index("date").resample("W-MON")["new_cases"].sum()
        plt.plot(df_d.index, df_d.values, label=d, linewidth=1.8)
    plt.title("EpiWatch — Weekly Disease Case Trends (Bhopal)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Weekly Cases")
    plt.legend(title="Disease")
    plt.savefig(output_dir / "disease_case_trends.png", dpi=300)
    plt.close()
    logger.info("Saved: disease_case_trends.png")

    # 2. Dynamic Per-Disease Time-Series Plots
    for d in diseases:
        d_clean = d.lower().replace("-", "_").replace(" ", "_")
        df_d = df[df["disease"] == d].sort_values("date")
        w_df = df_d.set_index("date").resample("W-MON")["new_cases"].sum().reset_index()

        # A. Weekly Cases
        plt.figure(figsize=(10, 4))
        plt.plot(w_df["date"], w_df["new_cases"], color="tab:blue", linewidth=1.5)
        plt.title(f"{d} — Historical Weekly Cases", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Weekly Cases")
        plt.savefig(output_dir / f"{d_clean}_weekly_cases.png", dpi=300)
        plt.close()

        # B. Weekly Cases + Moving Average
        w_df["ma_4w"] = w_df["new_cases"].rolling(4, min_periods=1).mean()
        w_df["ma_8w"] = w_df["new_cases"].rolling(8, min_periods=1).mean()
        w_df["ma_12w"] = w_df["new_cases"].rolling(12, min_periods=1).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(w_df["date"], w_df["new_cases"], alpha=0.3, label="Raw Weekly Cases", color="gray")
        plt.plot(w_df["date"], w_df["ma_4w"], label="4-Week MA", linewidth=1.8)
        plt.plot(w_df["date"], w_df["ma_8w"], label="8-Week MA", linewidth=1.8)
        plt.plot(w_df["date"], w_df["ma_12w"], label="12-Week MA", linewidth=1.8)
        plt.title(f"{d} — Moving Averages Trend Analysis", fontsize=12, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Cases")
        plt.legend()
        plt.savefig(output_dir / f"{d_clean}_moving_average.png", dpi=300)
        plt.close()

        # C. Monthly Cases
        m_df = df_d.set_index("date").resample("ME")["new_cases"].sum().reset_index()
        plt.figure(figsize=(10, 4))
        plt.bar(m_df["date"].dt.strftime("%Y-%m"), m_df["new_cases"], color="teal", alpha=0.8)
        plt.title(f"{d} — Monthly Cases Aggregation", fontsize=12, fontweight="bold")
        plt.xlabel("Month")
        plt.ylabel("Total Cases")
        plt.xticks(rotation=45)
        plt.savefig(output_dir / f"{d_clean}_monthly_cases.png", dpi=300)
        plt.close()

        # D. Yearly Cases
        y_df = df_d.set_index("date").resample("YE")["new_cases"].sum().reset_index()
        plt.figure(figsize=(6, 4))
        plt.bar(y_df["date"].dt.year.astype(str), y_df["new_cases"], color="darkorange", alpha=0.85)
        plt.title(f"{d} — Yearly Cases Comparison", fontsize=12, fontweight="bold")
        plt.xlabel("Year")
        plt.ylabel("Total Cases")
        plt.savefig(output_dir / f"{d_clean}_yearly_cases.png", dpi=300)
        plt.close()

    # 3. Combined Moving Averages
    plt.figure(figsize=(12, 6))
    for d in diseases:
        ts_d = ts_moving_avg[ts_moving_avg["disease"] == d]
        plt.plot(pd.to_datetime(ts_d["date"]), ts_d["cases_4w_ma"], label=f"{d} (4w MA)")
    plt.title("Combined Disease 4-Week Moving Averages", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("4-Week Moving Avg Cases")
    plt.legend()
    plt.savefig(output_dir / "moving_averages.png", dpi=300)
    plt.close()
    logger.info("Saved: moving_averages.png")

    # 4. Monthly & Yearly Trends
    m_agg_all = df.set_index("date").groupby([pd.Grouper(freq="ME"), "disease"])["new_cases"].sum().unstack()
    plt.figure(figsize=(12, 6))
    for col in m_agg_all.columns:
        plt.plot(m_agg_all.index, m_agg_all[col], label=col, linewidth=2.0)
    plt.title("Monthly Aggregate Case Trends Across Diseases", fontsize=14, fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Monthly Cases")
    plt.legend(title="Disease")
    plt.savefig(output_dir / "monthly_trends.png", dpi=300)
    plt.close()
    logger.info("Saved: monthly_trends.png")

    y_agg_all = df.set_index("date").groupby([pd.Grouper(freq="YE"), "disease"])["new_cases"].sum().unstack()
    plt.figure(figsize=(10, 5))
    y_agg_all.plot(kind="bar", stacked=False, figsize=(10, 5), ax=plt.gca())
    plt.title("Yearly Total Case Comparison by Disease", fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Total Cases")
    plt.xticks(range(len(y_agg_all.index)), [d.year for d in y_agg_all.index], rotation=0)
    plt.legend(title="Disease")
    plt.savefig(output_dir / "yearly_trends.png", dpi=300)
    plt.close()
    logger.info("Saved: yearly_trends.png")

    # 5. Seasonality Plot
    plt.figure(figsize=(10, 5))
    for d in diseases:
        s_d = season_df[season_df["disease"] == d]
        plt.plot(s_d["month"], s_d["avg_cases"], marker="o", label=d, linewidth=2)
    plt.title("Disease Seasonality — Average Cases by Month", fontsize=14, fontweight="bold")
    plt.xlabel("Month (1-12)")
    plt.ylabel("Average Daily Cases")
    plt.xticks(range(1, 13))
    plt.legend(title="Disease")
    plt.savefig(output_dir / "disease_seasonality.png", dpi=300)
    plt.close()
    logger.info("Saved: disease_seasonality.png")

    # 6. Growth Rates Distribution
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=growth_df[growth_df["period_weeks"] == 4], x="disease", y="growth_rate_percent")
    plt.title("Distribution of 4-Week Case Growth Rates (%) by Disease", fontsize=14, fontweight="bold")
    plt.xlabel("Disease")
    plt.ylabel("4-Week Growth Rate (%)")
    plt.axhline(0, color="gray", linestyle="--")
    plt.savefig(output_dir / "growth_rates.png", dpi=300)
    plt.close()
    logger.info("Saved: growth_rates.png")

    # 7. Incidence Trends
    inc_w = inc_df[inc_df["period"] == "weekly"].copy()
    inc_w["date"] = pd.to_datetime(inc_w["date"])
    plt.figure(figsize=(12, 6))
    for d in diseases:
        inc_d = inc_w[inc_w["disease"] == d]
        plt.plot(inc_d["date"], inc_d["incidence_per_100k"], label=d, linewidth=1.8)
    plt.title("Weekly Disease Incidence per 100,000 Population (Bhopal)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Incidence per 100k")
    plt.legend(title="Disease")
    plt.savefig(output_dir / "incidence_trends.png", dpi=300)
    plt.close()
    logger.info("Saved: incidence_trends.png")

    # 8. Positivity Over Time
    if not pos_df.empty and "positivity_rate_percent" in pos_df.columns:
        pos_df["date"] = pd.to_datetime(pos_df["month"])
        plt.figure(figsize=(12, 5))
        for d in diseases:
            p_d = pos_df[pos_df["disease"] == d]
            if not p_d.empty:
                plt.plot(p_d["date"], p_d["positivity_rate_percent"], marker="s", label=d)
        plt.title("Monthly Test Positivity Rate (%) Over Time", fontsize=14, fontweight="bold")
        plt.xlabel("Month")
        plt.ylabel("Test Positivity Rate (%)")
        plt.legend(title="Disease")
        plt.savefig(output_dir / "positivity_over_time.png", dpi=300)
        plt.close()
        logger.info("Saved: positivity_over_time.png")

    # 9. Cases vs Hospitalizations & Hospitalizations vs Deaths
    if not hosp_df.empty:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=hosp_df, x="cases", y="hospitalizations", hue="disease", s=60, alpha=0.8)
        plt.title("Monthly New Cases vs Hospitalizations", fontsize=14, fontweight="bold")
        plt.xlabel("Monthly Cases")
        plt.ylabel("Monthly Hospitalizations")
        plt.savefig(output_dir / "cases_vs_hospitalizations.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=hosp_df, x="hospitalizations", y="deaths", hue="disease", s=60, alpha=0.8)
        plt.title("Monthly Hospitalizations vs Deaths", fontsize=14, fontweight="bold")
        plt.xlabel("Monthly Hospitalizations")
        plt.ylabel("Monthly Deaths")
        plt.savefig(output_dir / "hospitalizations_vs_deaths.png", dpi=300)
        plt.close()
        logger.info("Saved: cases_vs_hospitalizations.png & hospitalizations_vs_deaths.png")

    # 10. Potential Statistical Anomalies
    plt.figure(figsize=(12, 6))
    anom_summary = anom_df.groupby(["disease", "potential_anomaly"]).size().unstack(fill_value=0)
    anom_summary.plot(kind="bar", stacked=False, figsize=(10, 5), color=["tab:blue", "tab:red"], ax=plt.gca())
    plt.title("Count of Normal vs Potential Statistical Anomaly Days per Disease", fontsize=14, fontweight="bold")
    plt.xlabel("Disease")
    plt.ylabel("Days Count")
    plt.legend(["Normal (0)", "Potential Anomaly (1)"])
    plt.xticks(rotation=0)
    plt.savefig(output_dir / "potential_anomalies.png", dpi=300)
    plt.close()
    logger.info("Saved: potential_anomalies.png")
