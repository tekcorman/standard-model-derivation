# /explainer/assets/

Media assets for the explainer site. Land into the appropriate subdirectory; reference from markdown via `assets/<subdir>/<file>`.

## Conventions

```
assets/
├── animations/   # MP4 / WebM / GIF — pre-rendered (Manim, Blender, etc.)
├── images/       # PNG / SVG — static diagrams, screenshots
├── 3d/           # Three.js scenes, model files (.gltf, .obj)
└── js/           # Custom JavaScript (math helpers, interactive widgets)
```

## Naming

- Snake_case lowercase: `f_inv_tree_growth.mp4`, `srs_z_cover_unfold.mp4`.
- Date suffix for sequenced revisions: `mdl_waterline_2026-05-26.svg`.
- Use SVG for diagrams (scales cleanly); use PNG only for raster screenshots or photos.

## Animation pipeline (recommended)

Manim is the recommended renderer for math/physics animations (this is what 3Blue1Brown uses). To render:

```bash
pip install manim
manim -qh scene.py SceneName    # high-quality MP4
manim -qm scene.py SceneName    # medium-quality preview
```

Output lands in `media/videos/scene/<quality>/`. Move the produced MP4 here, then reference from the relevant markdown page:

```markdown
![F_inv(E) tree growth](../assets/animations/f_inv_tree_growth.mp4)
```

(MkDocs Material renders mp4 inline via the `<video>` tag; works in any browser.)

## Animation priority queue (from the scaffolding plan)

1. **F_inv(E) Cayley tree growing** from a single toggle — for `story/01-what-can-exist.md`
2. **MDL waterline** with bars above/below — for `story/04-recurrence-and-the-mdl-waterline.md`
3. **srs → srs-z cover unfolding** with bipartite coloring — for `story/03-the-cover-that-holds-chirality.md`
4. **V_cb girth-cycle windings** as animated geometric series — for `story/06-the-two-pillars.md` or `over-determination.md`
5. **12-observable over-determination** as simultaneous channel readouts — for `over-determination.md` and `story/08-the-12-observable-overdetermination.md`

Each Manim scene + render is approximately 1-3 days of work depending on complexity. Pre-rendered MP4 embeds need no framework change.

## Interactive 3D

For the srs / srs-z 3D viewer (`story/03-the-cover-that-holds-chirality.md`), Three.js is the recommended option. Authoring approach when ready:

1. Build the scene as a standalone HTML file in `assets/3d/srs_viewer.html`.
2. Load it via an iframe from the markdown page, or via MkDocs Material's custom partials.
3. Decision point at that time: if interactive content grows substantially, evaluate switching to Docusaurus (better MDX/React interactivity story).

For now, this directory is a placeholder structure. Files land progressively as they're produced.
