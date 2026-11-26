# MegaLM concept brand kit

**Unofficial proposal only** — not an official TensorPlay asset. Prepared 2026-11-23.
The rationale, naming cautions and usage rules are in [`BRAND-GUIDELINES.md`](BRAND-GUIDELINES.md);
implementation tokens are in [`design-tokens.css`](design-tokens.css).

`asset-preview.png` is a quick visual index of every lockup. The SVGs under
`assets/` are the source of truth; open one directly in a browser, or import it
into Figma / Illustrator. PNGs are renders of those SVGs.

## Inventory

### Logo lockups (SVG + PNG)

| File stem | Background | Notes |
|---|---|---|
| `logo-horizontal-on-dark` | dark | Full lockup, white wordmark |
| `logo-horizontal-on-light` | light | Full lockup, navy wordmark |
| `logo-horizontal-mono-navy` | light | Single-colour horizontal |
| `logo-horizontal-mono-white` | dark | Single-colour horizontal |
| `logo-stacked-on-dark` | dark | Symbol above wordmark |
| `logo-stacked-on-light` | light | Symbol above wordmark |
| `wordmark-on-dark` | dark | Text only |
| `wordmark-on-light` | light | Text only |

### Symbol (SVG + PNG)

- `logo-symbol-color` — full colour, transparent background (this is the
  "on light" symbol; no separate alias copy is kept)
- `logo-symbol-white` — for dark surfaces / print
- `logo-symbol-navy` — for light surfaces / print

### Icons

- `app-icon-1024.svg` / `app-icon-1024.png` — navy rounded tile, full mark
- `favicon.svg` — simplified two-stream mark at 128 px
- `favicon-*.png` — raster sizes at 16, 32, 48, 64, 128 and 256 px
- `favicon.ico` — multi-resolution (16 / 32 / 48 / 64 / 128 / 256)

### Banners and wallpapers

- `github-banner-1280x320.svg` / `.png`
- `social-og-1200x630.svg` / `.png`
- `wallpaper-1920x1080.svg` / `.png`
- `wallpaper-1248x702.png` — legacy size from the original concept package

### Reference

- `color-system.svg` / `.png` — palette sheet
- `concept-brand-board.png` — AI-generated moodboard; the terminal copy in it
  is **illustrative, not real product output**

## Notes

- Each asset has exactly one canonical filename; there are no alias copies of
  the same artwork.
- The mono and stacked-light variants, the wordmark rasters, the 256 px favicon,
  `.ico`, the 1920×1080 wallpaper PNG and this preview sheet were rendered from
  the original SVGs, so scaling stays crisp and editable.
- Rendered text falls back to a system sans-serif when Inter is not installed;
  install Inter for the intended typography before final production use.
