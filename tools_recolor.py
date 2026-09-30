"""Recolours static/css/style.css from the blue theme to the beige / terracotta theme.

Blue accents become terracotta orange, navy text becomes dark ink, cool greys become
warm greys and light blue-grey backgrounds become cream. Reds, greens and ambers
(verdict colours) are kept.  The blue original is kept in old_version/style_blue.css.
"""
import colorsys
import re

SRC = "old_version/style_blue.css"
DST = "static/css/style.css"


def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def to_hex(rgb):
    return "#" + "".join(f"{round(max(0, min(1, c)) * 255):02x}" for c in rgb)


def recolor(rgb):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    hue = h * 360
    bluish = 190 <= hue <= 250 and s > 0.05
    if not bluish:
        return rgb                                   # whites, reds, greens, ambers stay
    if s > 0.45 and 0.25 < l < 0.85:                 # accent blues -> terracotta
        return colorsys.hls_to_rgb(11 / 360, min(l + 0.05, 0.9), 0.56)
    if l >= 0.9:                                     # very light tints -> cream
        return colorsys.hls_to_rgb(40 / 360, max(l - 0.02, 0.9), 0.35)
    if l >= 0.78:                                    # borders -> warm sand
        return colorsys.hls_to_rgb(40 / 360, l - 0.03, 0.2)
    if l < 0.3:                                      # navy -> dark ink
        return colorsys.hls_to_rgb(150 / 360, l, 0.14)
    return colorsys.hls_to_rgb(120 / 360, l, 0.05)   # slate greys -> warm grey


css = open(SRC, encoding="utf-8").read()
css = re.sub(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", lambda m: to_hex(recolor(hex_to_rgb(m.group(0)))), css)


def rgba(m):
    r, g, b = (int(x) / 255 for x in m.group(1, 2, 3))
    nr, ng, nb = recolor((r, g, b))
    return f"rgba({round(nr * 255)}, {round(ng * 255)}, {round(nb * 255)}, {m.group(4)})"


css = re.sub(r"rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\s*\)", rgba, css)

# page and navigation background, font
css = css.replace("font-family: Arial, Helvetica, sans-serif;", "font-family: 'Inter', Arial, Helvetica, sans-serif;")
open(DST, "w", encoding="utf-8").write(css)
print("recoloured", DST)
