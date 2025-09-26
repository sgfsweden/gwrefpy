import logging

logger = logging.getLogger(__name__)

# List of default colors
DEFAULT_COLORS = [
    "#4E79A7",  # deep blue
    "#F28E2B",  # orange
    "#E15759",  # red
    "#76B7B2",  # teal
    "#59A14F",  # green
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
    "#A0CBE8",  # light blue
    "#FFBE7D",  # light orange
    "#FFB7B2",  # light red
    "#CFCFCF",  # silver
    "#8CD17D",  # mint
    "#B6992D",  # gold
    "#F1CE63",  # light yellow
    "#499894",  # blue-green
    "#D37295",  # magenta
    "#FABFD2",  # light pink
]

# List of default monochrome colors (grayscale)
DEFAULT_MONOCHROME_COLORS = [
    "#000000",  # black
    "#444444",  # dark gray
    "#888888",  # gray
    "#BBBBBB",  # light gray
    "#DDDDDD",  # lighter gray
    "#E8E8E8",  # white
]

# List of default line styles
DEFAULT_LINE_STYLES = [
    "-",  # solid
    "--",  # dashed
    ":",  # dotted
    "-.",  # dashdot
]

# List of default marker styles
DEFAULT_MARKER_STYLES = [
    "o",  # circle
    "s",  # square
    "^",  # triangle up
    "v",  # triangle down
    "D",  # diamond
    "x",  # x
    "+",  # plus
    "*",  # star
    "p",  # pentagon
    "h",  # hexagon
]

# Dict for default plot attributes
DEFAULT_PLOT_ATTRIBUTES = {
    "color": None,
    "alpha": 1.0,
    "linestyle": None,
    "linewidth": 1.0,
    "marker": None,
    "markersize": 6,
    "marker_visible": False,
}

# Set the font size and family for matplotlib
FONT_SIZE = 14
FONT_FAMILY = "Times New Roman"
tfont = {"family": FONT_FAMILY, "size": FONT_SIZE}
afont = {"family": FONT_FAMILY, "size": FONT_SIZE - 2}
lfont = {"family": FONT_FAMILY, "size": FONT_SIZE - 2}
tifont = {"family": FONT_FAMILY, "size": FONT_SIZE - 3}


def print_constants():
    """
    Function that prints all predefined constants of the gwrefpy pacakge to the console.
    """
    logger.info("Default Colors: %s", DEFAULT_COLORS)
    logger.info("Default Monochrome Colors: %s", DEFAULT_MONOCHROME_COLORS)
    logger.info("Default Line Styles: %s", DEFAULT_LINE_STYLES)
    logger.info("Default Marker Styles: %s", DEFAULT_MARKER_STYLES)
    logger.info("Default Plot Attributes: %s", DEFAULT_PLOT_ATTRIBUTES)
    logger.info("Font Size: %s", FONT_SIZE)
    logger.info("Font Family: %s", FONT_FAMILY)
    logger.info("Title Font: %s", tfont)
    logger.info("Axis Font: %s", afont)
    logger.info("Label Font: %s", lfont)
    logger.info("Tick Font: %s", tifont)
