# MegaLM | Proposed Brand Identity v0.1

> Design proposal prepared 2026-11-23, **not an official TensorPlay asset**. The product classification “inference engine” is supplied by the requester; no public engine repository or performance figures were verified in the research pass.

## Research grounding

- **TensorPlay**: transparent deep-learning framework, C++/Python, CUDA/ROCm, with the line “See the system. Change it.” Sources: https://tensorplay.org and https://tensorplay.org/projects/.
- **MEGA / MEGA Hub**: open model, dataset, and AI application hub in the same ecosystem; official M symbol uses a green–yellow–orange orb/ribbon. Sources: https://mega.tensorplay.cn and https://tensorplay.org/projects/.
- **Organization**: https://github.com/ohtensorplay (GitHub displays “TensorPlay” and “MEGA”).
- **Caution**: MEGA Hub’s public home currently displays a **TensorPlay/MegaLM-3B** model entry, not verification of a standalone **MegaLM inference engine**. The engine is treated as an intended product from the requester’s description. Do not conflate a model name with the runtime.
- The organization’s logos are reference material only, **not re-created or modified** in this package. All MegaLM marks here are new concept artwork.

## Brand architecture

**TensorPlay** (parent / research & framework) → **MEGA Hub** (model/data/app platform) **and** **MegaLM** (inference engine, requested positioning). Treat MegaLM as an ecosystem product alongside MEGA Hub until an official technical integration or hierarchy is documented.

Use the name **MegaLM** (capital M + LM), and the endorsement **by TensorPlay**. Do not brand the inference engine as “MEGA Hub”, nor treat its proposed logo as the existing official M hub symbol.

## Core idea — “Token Stream M”

The mark is built from **three parallel path segments**: cyan (scheduling / incoming requests), citron (flow / execution), orange (emitted outputs). Their stepped trajectories form an abstract “M” while keeping a clear silhouette at reduced sizes. This is a visual metaphor, **not a technical architecture claim**.

### Positioning

- Category: Inference engine.
- Voice: precise, open, assured, infrastructure-native. No undocumented speed/throughput promises.
- Primary tagline: **Inference in motion.**
- Secondary description: **An inference engine in the TensorPlay ecosystem.**
- Do **not** put “fastest”, “3× faster”, “production-grade”, or unverified compatibility/support claims into the identity until benchmarks/docs exist.

## Colors

| Token | HEX | Function |
|---|---|---|
| Midnight Navy | `#07131B` | Hero and dark infrastructure surfaces |
| Vivid Cyan | `#18BFD1` | Primary brand signal; matches TensorPlay’s cyan heritage |
| Electric Citron | `#B4EC62` | Highlights and output focus; relates visually to MEGA’s greens |
| Warm Orange | `#FF995B` | Small secondary accents; echoes MEGA orange |
| Snow White | `#F5F8FA` | Main text and light surfaces |
| Muted Steel | `#9BACBA` | Secondary text on dark surfaces |

**Usage ratio:** navy/neutral 75%, cyan 15%, citron 8%, orange 2%. Never fill large blocks of the entire background with all three accent colors. Keep typography calm; the token streams should carry the motion.

## Typography and visual system

- Display: **Inter Display**, bold / extra-bold, tracking -0.02 to -0.04em.
- Body: **Inter**, regular to semibold.
- Code: system monospace (Menlo / Consolas / ui-monospace); keep code examples real or clearly marked as placeholders.
- Cards: 16px radius; buttons: 10px radius; 1px subdued outline and restrained blur on dark cards.
- Illustrations: thin compute grids; three offset stream lines; sparse square tokens indicating asynchronous output flow. Avoid stock brains, lightning bolts and processor-chip silhouettes.

## Logo use and accessibility

- Primary application: `assets/logo-horizontal-on-dark.svg` on dark backgrounds and `...on-light.svg` on light backgrounds.
- Compact profile: `assets/app-icon-1024.svg`, `assets/favicon.svg`.
- Monochrome: use `logo-symbol-white.svg` or `logo-symbol-navy.svg` for simple surfaces or print.
- Minimum symbol size: 32 px; use simplified `favicon.svg` at 32 px or smaller.
- Full horizontal logo: aim >= 240 px digital width. Minimum clear space = ~0.3× the icon width around the outer path.
- Preserve path proportions; do not skew. Do not apply shadows, photographs, or gradients inside the mark.
- Check contrast before introducing a custom shade; cyan/citron are primarily for icons/decoration, not extended body copy on white.

## Deliverables

- Editable vector logos: full-color symbol; 2 single-color versions; horizontal dark/light; wordmarks; stacked; app icon; responsive favicon.
- Ready raster assets: 1024 px app icon; 16 / 32 / 48 / 64 / 128 px favicons; GitHub banner 1280×320; Open Graph/social card 1200×630; wallpaper; palette sheet; selected logo previews.
- `design-tokens.css` for implementation.
- `concept-brand-board.png` — a separate **AI-generated visual moodboard**; visual copy in its example terminal is **illustrative, not actual product output**. The vector files in `assets/` are the production-friendly source of truth.

## Future verification before release

1. Confirm MegaLM’s public engine repo / docs and legal naming; check name uniqueness and potential confusion with the public “MegaLM-3B” model example.
2. Obtain the team’s approval for the brand architecture and mark.
3. Validate SVG rendering in current MEGA UI, README, GitHub social cards, favicon displays and dark-mode navigation.
4. Publish an official, versioned repository for brand assets only after approval.
