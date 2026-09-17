# Underwater examples

Opt-in Akinci boundary particles (`sph_akinci_boundary=True`). Stock flotation is #2857; see `../../UNDERWATER.md`.

```bash
python examples/underwater/buoyancy_demo.py
python examples/underwater/buoyancy_demo.py --compare
```

| Script | Purpose |
|--------|---------|
| `buoyancy_demo.py` | Stock density check plus stable Akinci on/off comparison |

Use `particle_size ≈ body_size / 7` and `dt ≤ 5e-4` for stable WCSPH.
