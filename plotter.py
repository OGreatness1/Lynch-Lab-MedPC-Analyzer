import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

LYNCH_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel


# ────────────────────────────────────────────────
# Helper added for robustness — detects date column automatically
# ────────────────────────────────────────────────
def get_date_column(df: pd.DataFrame) -> str | None:
    """
    Helper: Automatically detect the most likely date/time column.
    Used to make plots robust across raw sessions and aggregated daily data.
    """
    possible = ["start_date", "first_session_time", "date", "end_date"]
    for col in possible:
        if col in df.columns:
            return col
    return None


def create_plot(data, x, y, title, hue=None, kind="bar", palette="Set1", style=None, markers=True):
    """Legacy matplotlib fallback — kept for ZIP compatibility."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    try:
        if kind == "bar":
            sns.barplot(data=data, x=x, y=y, hue=hue, errorbar=None, palette=palette)
        elif kind == "line":
            sns.lineplot(data=data, x=x, y=y, hue=hue, style=style, markers=markers, palette=palette)
            plt.xticks(rotation=45)
        elif kind == "scatter":
            sns.scatterplot(data=data, x=x, y=y, hue=hue, size=style, alpha=0.7, palette=palette)
        elif kind == "box":
            sns.boxplot(data=data, x=x, y=y, hue=hue, palette=palette)
            sns.stripplot(data=data, x=x, y=y, color='black', alpha=0.3)
    except Exception:
        pass
    plt.title(title)
    plt.tight_layout()
    buf = io.BytesIO()
    try:
        plt.savefig(buf, format="png", dpi=150)
    except Exception as e:
        # Catch the Done exception from Agg backend
        if "Done" in str(e) or "RendererAgg" in str(e):
            pass  # This is expected - rendering completed
        else:
            st.warning(f"Matplotlib save error: {e}")
    plt.close()
    buf.seek(0)
    return buf


def create_interactive_plot(data, x, y, title, hue=None, kind="line", color_discrete_sequence=LYNCH_COLORS):
    """Core interactive Plotly plot — unchanged except better empty handling."""
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data available - {title}")
        return fig

    hover_cols = data.columns.tolist()

    if kind == "line":
        fig = px.line(
            data, x=x, y=y, color=hue, title=title, markers=True,
            hover_data=hover_cols, color_discrete_sequence=color_discrete_sequence
        )
        fig.update_xaxes(tickangle=45)
    elif kind == "bar":
        fig = px.bar(
            data, x=x, y=y, color=hue, title=title, barmode="group",
            hover_data=hover_cols, color_discrete_sequence=color_discrete_sequence
        )
    elif kind == "scatter":
        fig = px.scatter(
            data, x=x, y=y, color=hue, title=title, opacity=0.7,
            hover_data=hover_cols, marginal_x="box", marginal_y="box",
            color_discrete_sequence=color_discrete_sequence
        )
    elif kind == "box":
        fig = px.box(
            data, x=x, y=y, color=hue, title=title, points="all",
            hover_data=hover_cols, color_discrete_sequence=color_discrete_sequence
        )
    else:
        fig = px.line(
            data, x=x, y=y, title=title,
            hover_data=hover_cols, color_discrete_sequence=color_discrete_sequence
        )

    fig.update_layout(
        template="plotly_white",
        height=600,
        legend_title_text=hue or "",
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title()
    )
    return fig


# ────────────────────────────────────────────────
# Specialized plots — now all robust with date column detection
# ────────────────────────────────────────────────

def create_efficiency_trend(daily: pd.DataFrame):
    if daily.empty or "total_active_presses" not in daily.columns:
        return go.Figure().update_layout(title="Efficiency Trend")

    date_col = get_date_column(daily)
    if date_col is None:
        return go.Figure().update_layout(title="Efficiency Trend - No Date Column")

    df = daily.copy()
    df["efficiency"] = df["total_infusions"] / (df["total_active_presses"] + 1)

    return create_interactive_plot(
        df,
        date_col,
        "efficiency",
        "Efficiency Trend (Rewards/Effort)",
        hue="canonical_subject",
        kind="line"
    )


def create_hourly_heatmap(hr: pd.DataFrame):
    if hr.empty or "infusion_events" not in hr.columns:
        return go.Figure().update_layout(title="Hourly Infusion Heatmap - No Data")

    pivot = hr.groupby(["canonical_subject", "hour"])["infusion_events"].sum().unstack(fill_value=0)

    fig = px.imshow(
        pivot,
        title="Hourly Infusion Heatmap",
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Hour of Session", y="Subject", color="Infusions")
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


def create_cumulative_plot(sess: pd.DataFrame):
    if sess.empty:
        return go.Figure().update_layout(title="Cumulative Infusions - No Data")

    date_col = get_date_column(sess)
    if date_col is None:
        return go.Figure().update_layout(title="Cumulative Infusions - No Date Column")

    sess = sess.sort_values(date_col).copy()
    sess["cumulative_infusions"] = sess.groupby("canonical_subject")["infusions"].cumsum()

    fig = px.line(
        sess,
        x=date_col,
        y="cumulative_infusions",
        color="canonical_subject",
        title="Cumulative Infusions Over Time",
        markers=True,
        hover_data=["infusions", "active_presses", "program_name", "gender"],
        color_discrete_sequence=LYNCH_COLORS
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Infusions",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            tickformat="%Y-%m-%d",
            tickangle=45,
            rangeslider_visible=True
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_discrimination_plot(sess: pd.DataFrame):
    if sess.empty or "active_presses" not in sess.columns:
        return go.Figure().update_layout(title="Active vs Inactive Discrimination - No Data")

    date_col = get_date_column(sess)
    if date_col is None:
        return go.Figure().update_layout(title="Discrimination Plot - No Date Column")

    df_melt = sess.melt(
        id_vars=[date_col, "canonical_subject"],
        value_vars=["active_presses", "inactive_presses"],
        var_name="Lever",
        value_name="Presses"
    )
    df_melt["Lever"] = df_melt["Lever"].str.replace("_presses", "").str.title()

    fig = px.bar(
        df_melt,
        x=date_col,
        y="Presses",
        color="Lever",
        barmode="group",
        facet_col="canonical_subject",
        facet_col_wrap=3,
        title="Lever Discrimination per Session"
    )
    fig.update_layout(template="plotly_white", height=600)
    return fig


def create_pr_breakpoint_plot(sess: pd.DataFrame):
    if sess.empty or "breakpoint" not in sess.columns or sess["breakpoint"].sum() == 0:
        return go.Figure().update_layout(title="Progressive Ratio Breakpoints (No PR Data)")

    date_col = get_date_column(sess)
    if date_col is None:
        return go.Figure().update_layout(title="PR Breakpoint Plot - No Date Column")

    return create_interactive_plot(
        sess,
        date_col,
        "breakpoint",
        "PR Breakpoint Evolution",
        hue="canonical_subject",
        kind="line"
    )


def create_response_rate_plot(sess: pd.DataFrame):
    if sess.empty or "active_presses" not in sess.columns:
        return go.Figure().update_layout(title="Response Rate (Presses/Hour) - No Data")

    date_col = get_date_column(sess)
    if date_col is None:
        return go.Figure().update_layout(title="Response Rate Plot - No Date Column")

    rate = sess.copy()
    rate["response_rate"] = rate["active_presses"] / ((rate["duration_sec"] / 3600) + 1e-6)

    return create_interactive_plot(
        rate,
        date_col,
        "response_rate",
        "Response Rate (Active Presses per Hour)",
        hue="canonical_subject",
        kind="line"
    )


def create_mean_sem_trajectory(daily: pd.DataFrame):
    """
    Cohort average trajectory (Mean ± SEM) — robust date column detection.
    Full formatting, hover, and error bands preserved.
    """
    if daily.empty or "total_infusions" not in daily.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available for Cohort Average Trajectory")
        return fig

    date_col = get_date_column(daily)
    if date_col is None:
        fig = go.Figure()
        fig.update_layout(title="Cohort Average Trajectory - No Date Column")
        return fig

    mean_df = daily.groupby(date_col)["total_infusions"].agg(["mean", "sem"]).reset_index()
    mean_df = mean_df.rename(columns={date_col: "date"})

    # Convert to datetime for better formatting (safety)
    mean_df["date"] = pd.to_datetime(mean_df["date"], errors="coerce")

    fig = go.Figure([
        go.Scatter(
            name="Cohort Mean",
            x=mean_df["date"],
            y=mean_df["mean"],
            mode="lines+markers",
            line=dict(color="rgb(31, 119, 180)"),
            marker=dict(size=8),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Mean Infusions: %{y:.2f}<extra></extra>"
        ),
        go.Scatter(
            name="Upper Bound",
            x=mean_df["date"],
            y=mean_df["mean"] + mean_df["sem"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ),
        go.Scatter(
            name="Lower Bound",
            x=mean_df["date"],
            y=mean_df["mean"] - mean_df["sem"],
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(31, 119, 180, 0.2)",
            fill="tonexty",
            showlegend=False,
            hoverinfo="skip"
        )
    ])

    fig.update_layout(
        title="Cohort Average Trajectory (Mean ± SEM)",
        xaxis_title="Date",
        yaxis_title="Infusions (Mean ± SEM)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            tickformat="%Y-%m-%d",
            tickangle=45,
            rangeslider_visible=True
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig