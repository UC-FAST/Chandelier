from chandelier.ui.themes import (
    DEFAULT_THEME_ID,
    THEMES,
    get_theme,
    theme_id_from_options,
)


def test_registry_and_fallback():
    assert set(THEMES) == {
        "dark_pro",
        "high_contrast",
        "industrial",
        "classic",
        "xp_luna",
        "win2000",
        "win7",
        "win10",
    }
    assert get_theme("dark_pro").id == "dark_pro"
    assert get_theme("classic").render_style == "classic"
    assert get_theme("xp_luna").render_style == "luna"
    assert get_theme("win2000").render_style == "win2000"
    assert get_theme("win7").render_style == "win7"
    assert get_theme("win10").render_style == "win10"
    assert get_theme("does_not_exist").id == DEFAULT_THEME_ID


def test_theme_colours_are_bgr_triples():
    for theme in THEMES.values():
        for field in (
            "background",
            "header_bg",
            "surface",
            "surface_alt",
            "border",
            "footer_bg",
            "text",
            "text_dim",
            "accent",
            "accent_text",
            "success",
            "warning",
            "danger",
        ):
            color = getattr(theme, field)
            assert len(color) == 3
            assert all(isinstance(channel, int) and 0 <= channel <= 255 for channel in color)


def test_legacy_dict_contains_renderer_keys():
    legacy = get_theme("dark_pro").legacy_dict()
    for key in (
        "background",
        "cursor",
        "text",
        "boolTrue",
        "boolFalse",
        "numeral",
        "msg",
        "option",
        "irq",
        "cursorDisable",
        "textDisable",
    ):
        assert key in legacy


def test_classic_theme_uses_original_palette():
    legacy = get_theme("classic").legacy_dict()
    assert legacy["background"] == (0, 0, 0)
    assert legacy["cursor"] == (180, 130, 70)       # steelblue
    assert legacy["text"] == (255, 255, 255)        # white
    assert legacy["boolTrue"] == (47, 107, 85)      # darkolivegreen
    assert legacy["boolFalse"] == (0, 0, 139)       # darkred
    assert legacy["numeral"] == (11, 134, 184)      # darkgoldenrod
    assert legacy["cursorDisable"] == (169, 169, 169)  # darkgray
    assert legacy["textDisable"] == (128, 128, 128)    # gray


def test_theme_id_from_options():
    tree = {
        "camera": {
            "0": {"title": "Root", "options": []},
            "display": {
                "title": "Display",
                "options": [
                    {
                        "id": "theme",
                        "value": {"content": "Industrial", "value": "industrial"},
                    }
                ],
            },
        }
    }
    assert theme_id_from_options(tree) == "industrial"
    assert theme_id_from_options({}) == DEFAULT_THEME_ID
