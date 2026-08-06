# App store submission kit

Everything in this folder is a ready-to-submit listing for one app store or
catalog. This guide assumes you have never contributed to someone else's
GitHub repository before.

## The 60-second version of "fork" and "PR"

Every store below keeps its catalog as a public GitHub repository. You can't
edit their repository directly — instead:

1. **Fork** = click the **Fork** button on their repository page. GitHub
   creates *your own personal copy* of their repository under your account.
   You can edit your copy freely.
2. **Add your files** to your copy (GitHub's website lets you do this in the
   browser — no git commands needed: **Add file → Upload files** or
   **Create new file**).
3. **Pull request (PR)** = a button that appears after you change your copy
   ("Contribute → Open pull request"). It asks the store's maintainers to
   pull your change into the real catalog. They review it, maybe request
   tweaks (you just edit your copy again — the PR updates itself), then
   merge it. Once merged, OpenScrub is in their store.

That's the whole process for every store below. Only the folder layout and
file names differ.

**Before submitting anywhere:** make sure the latest release's Docker images
have finished building (the arm64 image ships from v1.0.54's workflow run
onward — CasaOS/Umbrel/Runtipi list arm64 support and will test it).

---

## 1. winget (Windows) — `deploy/winget/`

Target repository: `microsoft/winget-pkgs`

**Easiest path — skip the manual files entirely.** On any Windows machine:

```
winget install wingetcreate
wingetcreate new https://github.com/austinmabry/OpenScrub/releases/download/v1.0.54/OpenScrub-Setup-1.0.54.exe
```

Answer its prompts (package id: `AustinMabry.OpenScrub`; copy descriptions
from `AustinMabry.OpenScrub.locale.en-US.yaml`). It computes the installer
hash, builds the manifests, and **opens the PR for you** (it will ask to log
in to GitHub the first time).

**Manual path:** fork `microsoft/winget-pkgs`, create folder
`manifests/a/AustinMabry/OpenScrub/1.0.54/`, upload the three YAML files
from `deploy/winget/`, and first replace `REPLACE_WITH_SHA256` in the
installer manifest with the real hash — on Windows:
`certutil -hashfile OpenScrub-Setup-1.0.54.exe SHA256`.

For future releases: `wingetcreate update AustinMabry.OpenScrub -u <new exe url> -v <version> --submit` — one command per release.

## 2. CasaOS — `deploy/casaos/`

Target repository: `IceWhaleTech/CasaOS-AppStore` (official, slower review)
or `bigbeartechworld/big-bear-casaos` (community, faster).

Fork it, create folder `Apps/OpenScrub/` (official store layout), upload
`docker-compose.yml`, plus copy `assets/icon-256.png` from this repository
as the icon file their README asks for, then open the PR. Their
CONTRIBUTING.md shows the exact folder layout — mirror an existing app.

## 3. Runtipi — `deploy/runtipi/openscrub/`

Target repository: `runtipi/runtipi-appstore`

Fork it, create `apps/openscrub/`, upload `config.json` and
`docker-compose.json`, then create `apps/openscrub/metadata/` and upload
`metadata-description.md` renamed to `description.md` plus
`assets/icon-512.png` from this repository renamed to `logo.jpg` (convert
to JPEG first, or use PNG if their guide now allows it). Open the PR.

## 4. TrueNAS Community Apps — `deploy/truenas/openscrub/`

Target repository: `truenas/apps`

`deploy/truenas/openscrub/` is the COMPLETE app in their real layout —
app.yaml, README.md, ix_values.yaml, questions.yaml,
templates/docker-compose.yaml (Jinja2), and
templates/test_values/basic-values.yaml. Fork `truenas/apps`, open the
fork in github.dev (press "." on the fork page), create
`ix-dev/community/openscrub/`, and paste each file in unchanged. The
lib_version/lib_version_hash in app.yaml pin their template library —
their CI recalculates the hash, so leave them as committed. Commit,
then Contribute → Open pull request (title: "Add OpenScrub (community
train)"), fill their app-addition template. CI test-deploys the app
with basic-values.yaml; reviewers actively help first-time submitters —
a few rounds is normal. Bump the image tag in ix_values.yaml and
app_version in app.yaml to the current release before submitting.

## 5. Umbrel — `deploy/umbrel/openscrub/`

Target repository: `getumbrel/umbrel-apps`

Fork it, create folder `openscrub/`, upload `umbrel-app.yml` and
`docker-compose.yml`, then open the PR. They require a gallery image set —
add 3–5 screenshots (1600×1000 or larger) of the editor, review cards, and a
render; reference them in the `gallery:` list. Umbrel runs many Raspberry
Pis: the manifest already sets performance expectations.

## 6. Portainer templates — `deploy/portainer/template.json`

Target repository: `Lissy93/portainer-templates` (widely-used aggregate).

Fork it, open their `templates.json`, and add the object from
`template.json` into the `templates` array (alphabetical order). Open the PR.

## 7. CapRover — `deploy/caprover/openscrub.yml`

Target repository: `caprover/one-click-apps`

Fork it, upload `openscrub.yml` into `public/v4/apps/`, add a 512×512 logo
PNG named `openscrub.png` into `public/v4/logos/` (use `assets/icon-512.png`),
and open the PR.

## 8. Coolify — `deploy/coolify/openscrub.yaml`

Target repository: `coollabsio/coolify`

Fork it, upload `openscrub.yaml` into `templates/compose/`, add
`assets/icon-256.png` as `public/svgs/openscrub.png`, and open the PR.

## 9. Unraid Community Apps — `templates/` (repo root)

**This repository is already registered with Community Apps** — the
root `ca_profile.xml` plus `templates/openscrub.xml` (CPU) and
`templates/openscrub-nvidia.xml` (CUDA) shipped the first two listings.
Adding another container is just adding another XML file to
`templates/`; the CA feed picks up changes to registered repositories
on its periodic rescan (typically within a few hours).

To sanity-check a new or edited template before it goes live, the
submission portal at <https://ca.unraid.net/submit> can re-scan the
repository on demand: it validates the XML, checks for duplicates, and
previews the listing exactly as users will see it.

**Template conventions in this repo** (keep new templates consistent):

- Unraid exposes the port directly, so templates keep OpenScrub's
  default self-signed HTTPS (`WebUI` is `https://[IP]:[PORT:8384]/`)
  and the port config explains the one-time certificate warning.
- GPU access: NVIDIA rides `<ExtraParams>--runtime=nvidia</ExtraParams>`
  (needs the Nvidia Driver plugin); Intel rides a removable
  `/dev/dri` **Device** config (no host driver needed — the image
  ships Intel's media driver). Both degrade loudly to CPU.
- Appdata maps `/root/.local/share/OpenScrub` →
  `/mnt/user/appdata/openscrub`, with the description warning that
  jobs contain the sensitive uploads.
- Every template carries the optional Access-token variable and the
  read-only `/media` mount + `OPENSCRUB_MEDIA_ROOT` pair.
- `TemplateURL` must point at the raw GitHub URL of that exact file;
  bump `<Date>`/`<Changes>` when editing a published template.

## 10. PikaPods (no files needed)

Go to https://www.pikapods.com and use their "Suggest an app" form. If
accepted, they host OpenScrub for paying users and share revenue with you.
Note in the suggestion that self-hosting is the recommended deployment and
their hosted version suits users who accept a trusted host.

---

## Notes that apply everywhere

- **HTTP vs HTTPS:** OpenScrub serves HTTPS with a self-signed certificate
  by default. Stores that put apps behind their own reverse proxy (Umbrel,
  Runtipi, CapRover, Coolify) get `--http` in the command line of these
  manifests — the platform provides TLS. Stores that expose the port
  directly (CasaOS, TrueNAS, Portainer) keep the default HTTPS.
- **GPU:** all manifests use the CPU image (`pharmhero/openscrub:latest`)
  for maximum compatibility. The descriptions point NVIDIA users at the
  `:cuda` image.
- **Data:** the only volume that must persist is
  `/root/.local/share/OpenScrub` (jobs, reports, downloaded models,
  settings). An optional read-only `/media` mount lets users scan
  server-side files in place.
- **Versions in manifests:** winget, Runtipi, Umbrel and TrueNAS manifests
  pin a version string — bump it when you submit if a newer release exists.
