# Ocean prototype: media

The contact sheets and videos of the ocean prototype (the 2.5D height-field
rung, docs/lift/plan.md) are not in git: about 97 MB of PNG, GIF and MP4.
The code is (mapping/ocean/, tools/ocean_sheets.py, tools/ocean_videos.py,
tests/test_ocean.py), squashed in from branch exp/ocean (tip 3569764).

**Where they live.** `results/ocean_media/sheets/` and
`results/ocean_media/videos/` in the main checkout (gitignored, like every
run directory). They are the files of exp/ocean 3569764 under docs/ocean/;
that branch still has them in its history (`git archive exp/ocean docs/ocean`).

**Regenerate.**

    python tools/ocean_sheets.py --out results/ocean_media/sheets
    python tools/ocean_videos.py --out results/ocean_media/videos

Pass `--out`: the tools' default is still `docs/ocean/{sheets,videos}`,
which would put the media back inside the tracked tree. Never commit them.
