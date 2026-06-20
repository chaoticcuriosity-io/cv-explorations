# ci/

`github-pages.yml` is the GitHub **Actions** workflow that renders the Quarto site
and deploys it to GitHub Pages on every push to `main`.

It is parked here (instead of `.github/workflows/`) because pushing files under
`.github/workflows/` requires the `workflow` OAuth scope, which the current `gh`
token lacks. Until that's granted, the site is published the branch-based way via
a locally-rendered `gh-pages` branch (same public URL).

## To switch to automated Actions publishing (from a desktop)

```bash
gh auth refresh -h github.com -s workflow
git mv ci/github-pages.yml .github/workflows/publish.yml
git commit -m "Enable automated Pages publishing via Actions"
git push
# Then: repo Settings → Pages → Source = GitHub Actions
```
