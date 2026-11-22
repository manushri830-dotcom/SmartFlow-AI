"""Streamlit dashboard for the SmartFlow AI traffic simulation."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from config import Direction, SimulationConfig
from dashboard_data import (
    comparison_metrics_frame,
    direction_csv_bytes,
    direction_metrics_frame,
    history_csv_bytes,
    metrics_csv_bytes,
)
from dashboard_theme import APP_CSS, apply_dark_theme
from metrics import MetricsComparison, evaluate_comparison
from plots import (
    plot_arrivals_departures,
    plot_average_queue_length,
    plot_average_waiting_time,
    plot_congestion_index,
    plot_direction_average_queue,
    plot_direction_queues_over_time,
    plot_max_queue_length,
    plot_max_waiting_time,
    plot_queue_over_time,
    plot_signal_states,
    plot_throughput,
)
from simulator import ComparisonResult, run_comparison


COMPARISON_KEY = "smartflow_comparison"
METRICS_KEY = "smartflow_metrics"
STATUS_KEY = "smartflow_status"
CONFIG_KEY = "smartflow_config"


st.set_page_config(
    page_title="SmartFlow AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
:root {
  --bg: #080D19;
  --sidebar: #0D1628;
  --card: #121F33;
  --elevated: #192A42;
  --cyan: #00D4FF;
  --purple: #A78BFA;
  --green: #22C55E;
  --yellow: #FACC15;
  --red: #EF4444;
  --text: #F8FAFC;
  --muted: #94A3B8;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: rgba(8,13,25,.82) !important; backdrop-filter: blur(12px); z-index: 50 !important; }
[data-testid="stToolbar"] { z-index: 60 !important; }
[data-testid="stAppViewContainer"] > .main { background:
  radial-gradient(circle at 82% 8%, rgba(0,212,255,.055), transparent 22%),
  linear-gradient(180deg, #080D19 0%, #090F1D 100%) !important; }
section[data-testid="stSidebar"] {
  background: var(--sidebar) !important;
  border-right: 1px solid rgba(148,163,184,.10) !important;
  padding: 1.05rem 0.8rem 1.2rem !important;
  width: 280px !important;
  min-width: 280px !important;
  max-width: 280px !important;
  flex: 0 0 280px !important;
  z-index: 100 !important;
}
[data-testid="stSidebarCollapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 999999 !important;
  pointer-events: auto !important;
}
section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"],
button[aria-label="Collapse sidebar"] {
  display: inline-flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 1000002 !important;
  pointer-events: auto !important;
}
button[aria-label="Expand sidebar"] {
  position: fixed !important;
  left: 8px !important;
  top: 52px !important;
  display: inline-flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 1000003 !important;
  pointer-events: auto !important;
}
[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 999999 !important;
  pointer-events: auto !important;
}
@media (max-width: 768px) {
  section[data-testid="stSidebar"] {
    width: min(280px, 86vw) !important;
    min-width: min(280px, 86vw) !important;
    max-width: min(280px, 86vw) !important;
  }
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  min-height: 48px;
  border: 1px solid transparent;
  border-radius: 8px;
  padding: 0.7rem 0.75rem;
  margin: 0.16rem 0;
  color: var(--muted);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(0,212,255,.15), rgba(167,139,250,.055));
  border-color: rgba(0,212,255,.28);
  color: var(--text);
  box-shadow: inset 3px 0 0 var(--cyan);
}
[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight: 650; }
.block-container { max-width: 1500px; padding-top: 1.3rem; padding-bottom: 2.5rem; }
h1, h2, h3 { color: var(--text) !important; letter-spacing: 0 !important; }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, var(--card), rgba(25,42,66,.72)) !important;
  border: 1px solid rgba(148,163,184,.12) !important;
  border-radius: 8px !important;
  padding: 17px 18px !important;
  min-height: 112px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.65rem !important; }
[data-testid="stMetricDelta"] { color: var(--muted) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid rgba(148,163,184,.12); }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; color: var(--muted); }
.stTabs [aria-selected="true"] { background: rgba(0,212,255,.07); color: var(--cyan) !important; }
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(135deg, #082C40, #102B46) !important;
  border: 1px solid rgba(0,212,255,.48) !important;
  border-radius: 8px !important;
  min-height: 46px;
  color: var(--text) !important;
  box-shadow: 0 8px 24px rgba(0,212,255,.06);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--cyan) !important;
  background: linear-gradient(135deg, #0B3A51, #163653) !important;
  color: var(--cyan) !important;
}
[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.12); border-radius: 8px; overflow: hidden; }
.sf-card, [class="sf-card"] {
  background: linear-gradient(145deg, rgba(18,31,51,.96), rgba(25,42,66,.82)) !important;
  border: 1px solid rgba(148,163,184,.12) !important;
  border-radius: 8px !important;
}
.command-header { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; margin-bottom:18px; }
.command-title { font-size:1.82rem; line-height:1.1; font-weight:800; color:var(--text); }
.command-subtitle { color:var(--muted); margin-top:7px; }
.status-cluster { display:flex; align-items:center; gap:10px; }
.status-pill { display:flex; align-items:center; gap:8px; border:1px solid rgba(34,197,94,.24);
  background:rgba(34,197,94,.07); color:#DFF8E8; border-radius:999px; padding:8px 12px; font-size:.82rem; }
.status-dot { width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 12px var(--green); }
.time-pill { border:1px solid rgba(148,163,184,.12); background:var(--card); color:var(--muted);
  border-radius:999px; padding:8px 12px; font-size:.82rem; }
.kpi-grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; margin:8px 0 18px; }
.kpi-card { min-height:118px; background:linear-gradient(145deg,var(--card),rgba(25,42,66,.78));
  border:1px solid rgba(148,163,184,.12); border-radius:8px; padding:14px 15px; position:relative; overflow:hidden; }
.kpi-card:after { content:""; position:absolute; width:72px; height:72px; right:-38px; top:-38px;
  border-radius:50%; background:rgba(0,212,255,.08); filter:blur(2px); }
.kpi-icon { width:30px; height:30px; border-radius:7px; display:flex; align-items:center; justify-content:center;
  background:rgba(0,212,255,.09); border:1px solid rgba(0,212,255,.18); color:var(--cyan); font-weight:800; margin-bottom:10px; }
.kpi-label { color:var(--muted); font-size:.76rem; font-weight:650; text-transform:uppercase; }
.kpi-value { color:var(--text); font-size:1.42rem; line-height:1.2; font-weight:800; margin-top:5px; }
.kpi-note { color:var(--muted); font-size:.72rem; margin-top:5px; }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media (max-width: 700px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .command-header { align-items:flex-start; } }
</style>
""", unsafe_allow_html=True)


def _get_comparison() -> Optional[ComparisonResult]:
    """Return the session comparison if one has been executed."""
    value = st.session_state.get(COMPARISON_KEY)
    return value if isinstance(value, ComparisonResult) else None


def _get_metrics(comparison: ComparisonResult) -> MetricsComparison:
    """Return cached metrics for a comparison, evaluating them once."""
    cached = st.session_state.get(METRICS_KEY)
    if isinstance(cached, MetricsComparison):
        fixed = cached.fixed_time
        adaptive = cached.adaptive
        if (
            fixed.controller_type is comparison.fixed_time.controller_type
            and adaptive.controller_type is comparison.adaptive.controller_type
            and fixed.steps_completed == comparison.fixed_time.steps_completed
        ):
            return cached

    result = evaluate_comparison(comparison)
    st.session_state[METRICS_KEY] = result
    return result


def _traffic_density_label(density: str) -> float:
    """Convert the dashboard density preset to a per-direction arrival rate."""
    presets = {
        "Light": 0.08,
        "Moderate": 0.15,
        "Heavy": 0.30,
    }
    if density not in presets:
        raise ValueError(f"unknown traffic density {density!r}")
    return presets[density]


def _render_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="sf-card"><div class="sf-section-heading">No results yet</div>'
        f'<span class="sf-muted">{message}</span></div>',
        unsafe_allow_html=True,
    )


def _render_command_header(title: str, subtitle: str, comparison: Optional[ComparisonResult] = None) -> None:
    """Render the dashboard header with real simulation status and time."""
    if comparison is None:
        status = "Awaiting simulation"
        time_text = "Time --:--"
    else:
        status = "Simulation complete"
        max_time = max(
            (len(comparison.fixed_time.history), len(comparison.adaptive.history)),
            default=0,
        )
        time_text = f"Current +{max_time - 1 if max_time else 0} s"

    st.markdown(
        f'<div class="command-header">'
        f'<div><div class="command-title">{title}</div>'
        f'<div class="command-subtitle">{subtitle}</div></div>'
        f'<div class="status-cluster"><div class="status-pill"><span class="status-dot"></span>{status}</div>'
        f'<div class="time-pill">{time_text}</div></div></div>',
        unsafe_allow_html=True,
    )


def _render_metric_comparison(metrics: MetricsComparison) -> None:
    """Render headline metrics from fixed-time and adaptive results."""
    fixed = metrics.fixed_time
    adaptive = metrics.adaptive

    if adaptive.congestion_index >= 0.35:
        congestion_status = "High"
    elif adaptive.congestion_index >= 0.15:
        congestion_status = "Moderate"
    else:
        congestion_status = "Low"

    cards = (
        ("V", "Vehicles arrived", f"{adaptive.total_arrived:,}", "Adaptive controller"),
        ("D", "Vehicles departed", f"{adaptive.total_departed:,}", "Adaptive controller"),
        ("W", "Average waiting", f"{adaptive.average_waiting_time:.2f} s", "Lower is better"),
        ("Q", "Average queue", f"{adaptive.average_queue_length:.2f}", "Vehicles per lane"),
        ("M", "Maximum queue", f"{adaptive.max_queue_length}", "Recorded peak"),
        ("C", "Congestion", congestion_status, f"Index {adaptive.congestion_index:.3f}"),
    )

    html = '<div class="kpi-grid">'
    for icon, label, value, note in cards:
        html += (
            '<div class="kpi-card">'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{note}</div></div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Fixed-Time average waiting", f"{fixed.average_waiting_time:.2f} s")
    col2.metric("Adaptive average waiting", f"{adaptive.average_waiting_time:.2f} s")


def _signal_lights(state: object) -> str:
    """Return three realistic signal lamps, active state sourced from history."""
    value = getattr(state, "value", str(state)).lower()
    lights = ("red", "yellow", "green")
    html = '<div class="signal-housing">'
    for light in lights:
        active = " active" if light == value else ""
        glow = {
            "red": "rgba(239,68,68,.95)",
            "yellow": "rgba(250,204,21,.95)",
            "green": "rgba(34,197,94,.95)",
        }[light]
        html += (
            f'<span class="signal-lamp {light}{active}" '
            f'style="--lamp-glow:{glow}"></span>'
        )
    return html + '</div>'


def _render_junction(comparison: ComparisonResult) -> None:
    """Render a functional four-way intersection from actual simulation history."""
    st.markdown('<div class="command-header"><div><div class="command-title">Traffic Junction</div><div class="command-subtitle">Live topology and recorded signal telemetry</div></div></div>', unsafe_allow_html=True)

    controller = st.segmented_control(
        "Controller view",
        options=("Fixed-Time", "Adaptive Multi-Agent"),
        default="Adaptive Multi-Agent",
        key="junction_controller",
    )
    result = comparison.fixed_time if controller == "Fixed-Time" else comparison.adaptive

    if not result.history:
        _render_empty_state("The selected simulation has no recorded history.")
        return

    step_index = st.slider(
        "Simulation timeline",
        min_value=0,
        max_value=len(result.history) - 1,
        value=min(len(result.history) - 1, st.session_state.get("junction_step", 0)),
        key="junction_step",
    )
    record = result.history[step_index]
    active_phase = "Idle" if record.active_phase is None else record.active_phase.value

    states = {
        Direction.NORTH: record.signals[Direction.NORTH],
        Direction.SOUTH: record.signals[Direction.SOUTH],
        Direction.EAST: record.signals[Direction.EAST],
        Direction.WEST: record.signals[Direction.WEST],
    }

    st.markdown("""
    <style>
    .junction-shell { display:grid; grid-template-columns:minmax(620px,1fr) 220px; gap:12px; margin:10px 0 18px; }
    .intersection { position:relative; height:540px; overflow:hidden; border-radius:8px;
      border:1px solid rgba(148,163,184,.15); background:
      radial-gradient(circle at 50% 50%,rgba(0,212,255,.045),transparent 12%),
      linear-gradient(135deg,#080D19,#0B1322); }
    .road-v { position:absolute; left:50%; transform:translateX(-50%); width:210px; height:100%;
      background:linear-gradient(90deg,#162236 0 49%,#26364F 49% 51%,#162236 51% 100%); }
    .road-h { position:absolute; top:50%; transform:translateY(-50%); width:100%; height:210px;
      background:linear-gradient(180deg,#162236 0 49%,#26364F 49% 51%,#162236 51% 100%); }
    .road-v:before,.road-h:before { content:""; position:absolute; inset:0; opacity:.72; }
    .road-v:before { background:repeating-linear-gradient(180deg,transparent 0 38px,#94A3B8 38px 43px,transparent 43px 72px); width:3px; left:50%; }
    .road-h:before { background:repeating-linear-gradient(90deg,transparent 0 38px,#94A3B8 38px 43px,transparent 43px 72px); height:3px; top:50%; }
    .junction-core { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); width:210px; height:210px;
      border:1px solid rgba(0,212,255,.20); background:#26364F; z-index:2; }
    .junction-core:before,.junction-core:after { content:""; position:absolute; inset:18px; border:1px solid rgba(248,250,252,.09); }
    .direction-panel { position:absolute; z-index:5; min-width:148px; background:rgba(18,31,51,.96);
      border:1px solid rgba(148,163,184,.18); border-radius:8px; padding:9px 11px; text-align:center; box-shadow:0 12px 30px rgba(0,0,0,.24); }
    .direction-north { top:18px; left:50%; transform:translateX(-50%); }
    .direction-south { bottom:18px; left:50%; transform:translateX(-50%); }
    .direction-east { right:16px; top:50%; transform:translateY(-50%); }
    .direction-west { left:16px; top:50%; transform:translateY(-50%); }
    .direction-name { color:#94A3B8; font-size:.68rem; letter-spacing:.12em; font-weight:800; }
    .queue-readout { color:#F8FAFC; font-size:1.15rem; font-weight:800; margin-top:2px; }
    .queue-label { color:#94A3B8; font-size:.68rem; }
    .signal-housing { display:flex; flex-direction:row; gap:5px; justify-content:center; margin:7px auto 3px;
      width:max-content; background:#080D19; border:1px solid #334155; border-radius:8px; padding:5px; }
    .signal-lamp { width:11px; height:11px; border-radius:50%; background:#26364F; opacity:.42; }
    .signal-lamp.active { opacity:1; background:var(--lamp-glow); box-shadow:0 0 11px var(--lamp-glow),0 0 22px var(--lamp-glow); }
    .telemetry { display:flex; flex-direction:column; gap:10px; }
    .telemetry-card { background:linear-gradient(145deg,#121F33,#192A42); border:1px solid rgba(148,163,184,.12);
      border-radius:8px; padding:13px; }
    .telemetry-label { color:#94A3B8; font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; }
    .telemetry-value { color:#F8FAFC; font-size:1.22rem; font-weight:800; margin-top:4px; }
    .telemetry-value.cyan { color:#00D4FF; }
    @media (max-width:900px) { .junction-shell { grid-template-columns:1fr; } .telemetry { display:grid; grid-template-columns:repeat(2,1fr); } }
    </style>
    """, unsafe_allow_html=True)

    def panel(direction: Direction, css_class: str) -> str:
        name = direction.value
        queue = record.queue_lengths[direction]
        return (
            f'<div class="direction-panel direction-{css_class}">'
            f'<div class="direction-name">{name.upper()}</div>'
            f'{_signal_lights(states[direction])}'
            f'<div class="queue-readout">{queue}</div>'
            f'<div class="queue-label">vehicles queued</div></div>'
        )

    junction_html = (
        '<div class="junction-shell"><div class="intersection">'
        '<div class="road-v"></div><div class="road-h"></div><div class="junction-core"></div>'
        + panel(Direction.NORTH, "north")
        + panel(Direction.SOUTH, "south")
        + panel(Direction.EAST, "east")
        + panel(Direction.WEST, "west")
        + '</div><div class="telemetry">'
        f'<div class="telemetry-card"><div class="telemetry-label">Simulation time</div><div class="telemetry-value cyan">+{record.time} s</div></div>'
        f'<div class="telemetry-card"><div class="telemetry-label">Stage</div><div class="telemetry-value">{record.stage.value.title()}</div></div>'
        f'<div class="telemetry-card"><div class="telemetry-label">Active phase</div><div class="telemetry-value">{active_phase}</div></div>'
        f'<div class="telemetry-card"><div class="telemetry-label">Total waiting</div><div class="telemetry-value">{record.total_waiting}</div></div>'
        '</div></div>'
    )
    st.markdown(junction_html, unsafe_allow_html=True)


def _render_chart(title: str, figure) -> None:
    """Apply the project theme and display a Matplotlib figure."""
    st.subheader(title)
    themed = apply_dark_theme(figure)
    st.pyplot(themed, clear_figure=True)


def render_overview(comparison: ComparisonResult, metrics: MetricsComparison) -> None:
    _render_command_header(
        "SMARTFLOW AI",
        "Intelligent Traffic Management System",
        comparison,
    )
    _render_metric_comparison(metrics)

    st.markdown('<div class="sf-section-heading">Controller Comparison</div>', unsafe_allow_html=True)
    fixed = metrics.fixed_time
    adaptive = metrics.adaptive
    table = comparison_metrics_frame(metrics)

    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.markdown('<div class="sf-card"><b>Fixed-Time</b><br><span class="sf-muted">Baseline strategy</span></div>', unsafe_allow_html=True)
        st.metric("Fixed-Time avg. waiting", f"{fixed.average_waiting_time:.2f} s")
    with col2:
        st.markdown('<div class="sf-card"><b>Adaptive Multi-Agent</b><br><span class="sf-muted">Queue and waiting-aware coordination</span></div>', unsafe_allow_html=True)
        st.metric("Adaptive avg. waiting", f"{adaptive.average_waiting_time:.2f} s")

    st.dataframe(table, width="stretch", hide_index=True)


def render_simulation_controls(comparison: Optional[ComparisonResult] = None) -> None:
    _render_command_header(
        "Simulation Controls",
        "Configure and execute a fair fixed/adaptive comparison",
        comparison,
    )
    st.markdown('<div class="sf-section-heading">Run Configuration</div>', unsafe_allow_html=True)

    duration = st.number_input("Duration (seconds)", min_value=1, max_value=86400, value=600, step=60)
    density = st.selectbox("Traffic density", ("Light", "Moderate", "Heavy"), index=1)
    seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, step=1)

    with st.expander("Per-direction arrival rates", expanded=False):
        default_rate = _traffic_density_label(density)
        north = st.number_input("North (vehicles/second)", 0.0, 10.0, default_rate, 0.01)
        south = st.number_input("South (vehicles/second)", 0.0, 10.0, default_rate, 0.01)
        east = st.number_input("East (vehicles/second)", 0.0, 10.0, default_rate, 0.01)
        west = st.number_input("West (vehicles/second)", 0.0, 10.0, default_rate, 0.01)

    if st.button("Run Fixed-Time & Adaptive Simulation", type="primary", width="stretch"):
        try:
            config = SimulationConfig(
                duration=int(duration),
                seed=int(seed),
                arrival_rate_north=float(north),
                arrival_rate_south=float(south),
                arrival_rate_east=float(east),
                arrival_rate_west=float(west),
            )
            with st.spinner("Running both controllers on the same traffic schedule..."):
                comparison = run_comparison(config)
            st.session_state[COMPARISON_KEY] = comparison
            st.session_state[METRICS_KEY] = evaluate_comparison(comparison)
            st.session_state[CONFIG_KEY] = config
            st.session_state[STATUS_KEY] = (
                f"Simulation complete: {comparison.fixed_time.steps_completed} seconds simulated."
            )
            st.success(st.session_state[STATUS_KEY])
        except (TypeError, ValueError, RuntimeError) as error:
            st.error(f"Simulation could not start: {error}")

    status = st.session_state.get(STATUS_KEY)
    if status:
        st.info(status)


def render_analytics(comparison: ComparisonResult, metrics: MetricsComparison) -> None:
    _render_command_header(
        "Performance Analytics",
        "Controller performance and network telemetry",
        comparison,
    )

    tab_names = ("Comparison", "Queue Trends", "Signals & Flow")
    comparison_tab, queue_tab, flow_tab = st.tabs(tab_names)

    with comparison_tab:
        col1, col2 = st.columns(2)
        with col1:
            _render_chart("Average Waiting Time", plot_average_waiting_time(metrics))
            _render_chart("Maximum Waiting Time", plot_max_waiting_time(metrics))
            _render_chart("Average Queue Length", plot_average_queue_length(metrics))
        with col2:
            _render_chart("Maximum Queue Length", plot_max_queue_length(metrics))
            _render_chart("Throughput", plot_throughput(metrics))
            _render_chart("Congestion Index", plot_congestion_index(metrics))

        _render_chart("Average Queue by Direction", plot_direction_average_queue(metrics))

    with queue_tab:
        _render_chart(
            "Queue Length Over Time",
            plot_queue_over_time(comparison.fixed_time, comparison.adaptive),
        )
        col1, col2 = st.columns(2)
        with col1:
            _render_chart("Fixed-Time Direction Queues", plot_direction_queues_over_time(comparison.fixed_time))
        with col2:
            _render_chart("Adaptive Direction Queues", plot_direction_queues_over_time(comparison.adaptive))

    with flow_tab:
        col1, col2 = st.columns(2)
        history_length = len(comparison.fixed_time.history)
        window_end = min(300, history_length)
        with col1:
            _render_chart(
                "Fixed-Time Signal States",
                plot_signal_states(comparison.fixed_time, 0, window_end if window_end > 0 else None),
            )
            _render_chart("Fixed-Time Arrivals & Departures", plot_arrivals_departures(comparison.fixed_time))
        with col2:
            _render_chart(
                "Adaptive Signal States",
                plot_signal_states(comparison.adaptive, 0, window_end if window_end > 0 else None),
            )
            _render_chart("Adaptive Arrivals & Departures", plot_arrivals_departures(comparison.adaptive))


def render_reports(comparison: ComparisonResult, metrics: MetricsComparison) -> None:
    _render_command_header(
        "Reports",
        "Verified simulation output and downloadable records",
        comparison,
    )

    st.subheader("Comparison Metrics")
    st.dataframe(comparison_metrics_frame(metrics), width="stretch", hide_index=True)

    st.subheader("Direction-wise Metrics")
    st.dataframe(direction_metrics_frame(metrics), width="stretch", hide_index=True)

    st.subheader("CSV Downloads")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "Download history CSV",
        data=history_csv_bytes(comparison.fixed_time),
        file_name="smartflow_fixed_history.csv",
        mime="text/csv",
        width="stretch",
    )
    col2.download_button(
        "Download metrics CSV",
        data=metrics_csv_bytes(metrics),
        file_name="smartflow_metrics.csv",
        mime="text/csv",
        width="stretch",
    )
    col3.download_button(
        "Download direction CSV",
        data=direction_csv_bytes(metrics),
        file_name="smartflow_direction_metrics.csv",
        mime="text/csv",
        width="stretch",
    )

    st.subheader("Simulation Summary")
    fixed = metrics.fixed_time
    adaptive = metrics.adaptive

    summary = (
        f'<div class="sf-card">'
        f'<b>Fixed-Time:</b> {fixed.total_arrived} arrived, {fixed.total_departed} departed, '
        f'{fixed.vehicles_remaining} remaining.<br>'
        f'<b>Adaptive:</b> {adaptive.total_arrived} arrived, {adaptive.total_departed} departed, '
        f'{adaptive.vehicles_remaining} remaining.<br>'
        f'<b>Average waiting:</b> fixed {fixed.average_waiting_time:.2f} s; '
        f'adaptive {adaptive.average_waiting_time:.2f} s.<br>'
        f'<b>Congestion index:</b> fixed {fixed.congestion_index:.3f}; '
        f'adaptive {adaptive.congestion_index:.3f}.'
        f'</div>'
    )
    st.markdown(summary, unsafe_allow_html=True)

    waiting_row = metrics.row("average_waiting_time")
    throughput_row = metrics.row("throughput_per_minute")
    if waiting_row.winner.value == "adaptive":
        waiting_finding = "The adaptive controller had the lower average waiting time."
    elif waiting_row.winner.value == "fixed_time":
        waiting_finding = "The fixed-time controller had the lower average waiting time."
    else:
        waiting_finding = "The controllers tied on average waiting time."

    if throughput_row.winner.value == "adaptive":
        throughput_finding = "The adaptive controller had the higher throughput."
    elif throughput_row.winner.value == "fixed_time":
        throughput_finding = "The fixed-time controller had the higher throughput."
    else:
        throughput_finding = "The controllers tied on throughput."

    st.markdown(
        f'<div class="sf-card"><b>Key findings</b><br>{waiting_finding}<br>{throughput_finding}</div>',
        unsafe_allow_html=True,
    )


# Sidebar navigation and configuration
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:5px 4px 2px">
      <div style="width:38px;height:38px;border-radius:8px;background:linear-gradient(135deg,#00D4FF,#A78BFA);display:flex;align-items:center;justify-content:center;color:#080D19;font-weight:900">S</div>
      <div><div style="font-size:1.18rem;font-weight:850;color:#F8FAFC">SMARTFLOW <span style="color:#00D4FF">AI</span></div><div style="color:#94A3B8;font-size:.72rem;margin-top:2px">Multi-Agent Traffic Intelligence</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        (
            "01  Overview",
            "02  Traffic Junction",
            "03  Simulation Controls",
            "04  Performance Analytics",
            "05  Reports",
        ),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="sf-section-heading">Simulation Configuration</div>', unsafe_allow_html=True)

comparison = _get_comparison()
metrics = _get_metrics(comparison) if comparison is not None else None

if page == "01  Overview":
    if comparison is None:
        _render_command_header("Overview", "Network operation summary", None)
        _render_empty_state("Run a simulation in Simulation Controls to populate these results.")
    else:
        render_overview(comparison, metrics)

elif page == "02  Traffic Junction":
    if comparison is None:
        _render_command_header("Traffic Junction", "Recorded signal and queue telemetry", None)
        _render_empty_state("Run a simulation to inspect recorded signal states and queues.")
    else:
        _render_junction(comparison)

elif page == "03  Simulation Controls":
    render_simulation_controls(comparison)

elif page == "04  Performance Analytics":
    if comparison is None:
        _render_command_header("Performance Analytics", "Controller performance and network telemetry", None)
        _render_empty_state("Run a simulation to generate the real comparison charts.")
    else:
        render_analytics(comparison, metrics)

elif page == "05  Reports":
    if comparison is None:
        _render_command_header("Reports", "Verified simulation output", None)
        _render_empty_state("Run a simulation to populate tables, downloads, and findings.")
    else:
        render_reports(comparison, metrics)
