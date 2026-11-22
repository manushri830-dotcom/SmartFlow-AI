"""Reusable visual theme helpers for the SmartFlow AI dashboard."""

from matplotlib.figure import Figure

BACKGROUND = "#0B1220"
CARD = "#172338"
CARD_ALT = "#1E2D45"
CYAN = "#00D4FF"
PURPLE = "#A78BFA"
GREEN = "#22C55E"
YELLOW = "#FACC15"
RED = "#EF4444"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
GRID = "#334155"

APP_CSS = """
<style>
.stApp {
  background: #0B1220;
  color: #F8FAFC;
}
[data-testid="stSidebar"] {
  background: #172338;
  border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
  color: #F8FAFC;
}
[data-testid="stMetric"] {
  background: #172338;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}
[data-testid="stMetricLabel"] {
  color: #94A3B8;
  font-weight: 600;
}
[data-testid="stMetricValue"] {
  color: #F8FAFC;
}
.stButton > button, .stDownloadButton > button {
  background: #172338;
  color: #F8FAFC;
  border: 1px solid #00D4FF;
  border-radius: 10px;
  padding: 0.55rem 1rem;
  font-weight: 600;
  transition: all 0.16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: #1E2D45;
  border-color: #A78BFA;
  color: #00D4FF;
}
h1, h2, h3 {
  color: #F8FAFC;
  letter-spacing: 0;
}
.sf-section-heading {
  color: #00D4FF;
  font-size: 1.1rem;
  font-weight: 700;
  margin: 1.1rem 0 0.65rem;
}
.sf-card {
  background: #172338;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
}
div[data-testid="stAlert"] {
  border-radius: 12px;
}
</style>
"""


def apply_dark_theme(fig: Figure) -> Figure:
    """Apply the SmartFlow AI dark theme to a Matplotlib figure in place.

    The function updates every axes in ``fig`` and returns the same Figure,
    allowing it to be used with figures created directly or through plots.py.
    """
    if not isinstance(fig, Figure):
        raise ValueError(f"fig must be a matplotlib.figure.Figure, got {fig!r}")

    fig.patch.set_facecolor(BACKGROUND)

    for ax in fig.axes:
        ax.set_facecolor(CARD)
        ax.title.set_color(TEXT)

        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=MUTED)

        for spine in ax.spines.values():
            spine.set_color(GRID)

        ax.set_axisbelow(True)
        ax.grid(True, color=GRID, alpha=0.45, linewidth=0.7)

        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(CARD_ALT)
            frame.set_edgecolor(GRID)
            for label in legend.get_texts():
                label.set_color(TEXT)

        for text in ax.texts:
            if text.get_color() in ("black", "#000000"):
                text.set_color(TEXT)

    fig.tight_layout()
    return fig

