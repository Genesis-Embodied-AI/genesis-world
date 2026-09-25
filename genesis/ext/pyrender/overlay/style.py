"""ImGui style configuration for the Genesis overlay panel."""


def apply_dark_theme(imgui) -> None:
    """Apply the modern rounded dark theme used by the Genesis Control Panel."""
    imgui.style_colors_dark()
    style = imgui.get_style()
    Col_ = imgui.Col_
    sc = style.set_color_

    # Geometry - modern rounded, borderless
    style.window_rounding = 12.0
    style.frame_rounding = 8.0
    style.child_rounding = 10.0
    style.popup_rounding = 10.0
    style.scrollbar_rounding = 8.0
    style.grab_rounding = 6.0
    style.tab_rounding = 8.0
    style.window_border_size = 0.0
    style.frame_border_size = 0.0

    # Spacing
    style.window_padding = (12.0, 10.0)
    style.frame_padding = (8.0, 5.0)
    style.item_spacing = (8.0, 6.0)
    style.item_inner_spacing = (6.0, 4.0)
    style.scrollbar_size = 10.0
    style.grab_min_size = 10.0

    # Semi-transparent backgrounds
    sc(Col_.window_bg, (0.11, 0.11, 0.14, 0.92))
    sc(Col_.child_bg, (0.13, 0.13, 0.16, 0.60))
    sc(Col_.popup_bg, (0.11, 0.11, 0.14, 0.96))

    # Text
    sc(Col_.text, (0.93, 0.94, 0.96, 1.0))
    sc(Col_.text_disabled, (0.45, 0.47, 0.52, 1.0))

    # Borders - subtle
    sc(Col_.border, (0.25, 0.26, 0.30, 0.35))

    # Frames (sliders, input fields) - frosted
    sc(Col_.frame_bg, (0.18, 0.18, 0.22, 0.75))
    sc(Col_.frame_bg_hovered, (0.24, 0.24, 0.30, 0.85))
    sc(Col_.frame_bg_active, (0.28, 0.28, 0.36, 0.95))

    # Title bar
    sc(Col_.title_bg, (0.09, 0.09, 0.12, 0.95))
    sc(Col_.title_bg_active, (0.12, 0.12, 0.16, 1.0))
    sc(Col_.title_bg_collapsed, (0.09, 0.09, 0.12, 0.70))

    # Buttons - accent blue with soft edges
    sc(Col_.button, (0.22, 0.38, 0.58, 0.80))
    sc(Col_.button_hovered, (0.28, 0.48, 0.70, 0.90))
    sc(Col_.button_active, (0.20, 0.34, 0.52, 1.0))

    # Headers (collapsing headers) - subtle highlight
    sc(Col_.header, (0.18, 0.18, 0.24, 0.65))
    sc(Col_.header_hovered, (0.26, 0.40, 0.58, 0.75))
    sc(Col_.header_active, (0.24, 0.38, 0.56, 0.90))

    # Interactive accents - bright blue
    sc(Col_.check_mark, (0.45, 0.72, 0.95, 1.0))
    sc(Col_.slider_grab, (0.38, 0.62, 0.88, 0.90))
    sc(Col_.slider_grab_active, (0.45, 0.72, 0.95, 1.0))

    # Scrollbar - minimal
    sc(Col_.scrollbar_bg, (0.08, 0.08, 0.10, 0.30))
    sc(Col_.scrollbar_grab, (0.30, 0.32, 0.38, 0.50))
    sc(Col_.scrollbar_grab_hovered, (0.40, 0.42, 0.50, 0.70))
    sc(Col_.scrollbar_grab_active, (0.48, 0.50, 0.58, 0.90))

    # Tabs
    sc(Col_.tab, (0.14, 0.14, 0.18, 0.70))
    sc(Col_.tab_hovered, (0.28, 0.46, 0.66, 0.85))
    sc(Col_.tab_selected, (0.22, 0.38, 0.58, 0.90))

    # Separators - very subtle
    sc(Col_.separator, (0.28, 0.30, 0.36, 0.30))
    sc(Col_.separator_hovered, (0.38, 0.56, 0.78, 0.60))
    sc(Col_.separator_active, (0.42, 0.64, 0.88, 0.85))

    # Resize grip
    sc(Col_.resize_grip, (0.28, 0.40, 0.58, 0.20))
    sc(Col_.resize_grip_hovered, (0.35, 0.55, 0.78, 0.50))
    sc(Col_.resize_grip_active, (0.40, 0.65, 0.90, 0.75))
