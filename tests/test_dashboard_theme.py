"""Tests for the SmartFlow AI dashboard theme helpers."""

import unittest

from matplotlib.figure import Figure

import dashboard_theme


class TestDashboardTheme(unittest.TestCase):
    def test_color_constants(self):
        expected = {
            "BACKGROUND": "#0B1220",
            "CARD": "#172338",
            "CARD_ALT": "#1E2D45",
            "CYAN": "#00D4FF",
            "PURPLE": "#A78BFA",
            "GREEN": "#22C55E",
            "YELLOW": "#FACC15",
            "RED": "#EF4444",
            "TEXT": "#F8FAFC",
            "MUTED": "#94A3B8",
            "GRID": "#334155",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(dashboard_theme, name), value)

    def test_app_css_contains_dashboard_sections(self):
        css = dashboard_theme.APP_CSS
        self.assertIsInstance(css, str)
        for required in (
            "#0B1220",
            "[data-testid=\"stSidebar\"]",
            "[data-testid=\"stMetric\"]",
            ".stButton > button",
            ".stDownloadButton > button",
            "h1, h2, h3",
            ".sf-card",
            "border-radius",
        ):
            with self.subTest(required=required):
                self.assertIn(required, css)

    def test_apply_dark_theme_styles_single_axis_and_returns_same_figure(self):
        figure = Figure(figsize=(6, 4))
        axes = figure.subplots()
        axes.set_title("Traffic Queue")
        axes.set_xlabel("Simulation time")
        axes.set_ylabel("Vehicles")
        axes.plot([0, 1], [2, 1], label="North")
        axes.legend()

        returned = dashboard_theme.apply_dark_theme(figure)

        self.assertIs(returned, figure)
        self.assertEqual(figure.patch.get_facecolor(), (11 / 255, 18 / 255, 32 / 255, 1.0))
        self.assertEqual(axes.get_facecolor(), (23 / 255, 35 / 255, 56 / 255, 1.0))
        self.assertEqual(axes.title.get_color(), dashboard_theme.TEXT)
        self.assertEqual(axes.xaxis.label.get_color(), dashboard_theme.MUTED)
        self.assertEqual(axes.yaxis.label.get_color(), dashboard_theme.MUTED)
        legend = axes.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(legend.get_texts()[0].get_color(), dashboard_theme.TEXT)

    def test_apply_dark_theme_styles_multiple_axes(self):
        figure = Figure(figsize=(8, 5))
        first = figure.add_subplot(1, 2, 1)
        second = figure.add_subplot(1, 2, 2)
        first.plot([0, 1], [0, 3])
        second.plot([0, 1], [3, 0])

        returned = dashboard_theme.apply_dark_theme(figure)

        self.assertIs(returned, figure)
        self.assertEqual(len(figure.axes), 2)
        for axes in figure.axes:
            self.assertEqual(axes.get_facecolor(), (23 / 255, 35 / 255, 56 / 255, 1.0))
            for spine in axes.spines.values():
                self.assertEqual(spine.get_edgecolor(), (51 / 255, 65 / 255, 85 / 255, 1.0))

    def test_apply_dark_theme_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "Figure"):
            dashboard_theme.apply_dark_theme(object())


if __name__ == "__main__":
    unittest.main()
