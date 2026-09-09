import seaborn as sns


BAR_COLOR = "#2C6E9E"
EDGE_COLOR = "#1A3F5C"
ACCENT_COLOR = "#8FB8D9"
ACCENT_EDGE = "#4A7A9E"
GRID_COLOR = "#CCCCCC"

LABEL_FS = 10
TITLE_FS = 14
AXIS_FS = 12

sns.set_theme(
    style="whitegrid",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
    },
)
