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

## 4. TrueNAS Community Apps — `deploy/truenas/`

Target repository: `truenas/apps`

This store wraps apps in their own template library, so the files here
(`app.yaml` + reference `docker-compose.yml`) are your *starting point*, not
the final layout. Open a PR with a new folder under `ix-dev/community/openscrub/`
mirroring a simple existing app (look at one like `actual-budget`), pasting
the metadata from `app.yaml`. Their reviewers actively help first-time
submitters shape it — it's normal for this one to take a few rounds.

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

## 9. Unraid Community Apps — `deploy/unraid/`

Target: the **Community Apps submission portal** at
<https://ca.unraid.net/submit> — Unraid replaced the old forum-thread
process with a self-service portal, so there is no fork-and-PR here.
Instead you host a small "template repository" on your own GitHub account
and register it once.

**Step 1 — create your template repository.** On GitHub, create a new
public repository named `unraid-templates` under your account (the
TemplateURL inside the XML assumes `austinmabry/unraid-templates` on
branch `main` — if you pick a different name, update that one line).
Unraid publishes a starter you can compare against:
<https://github.com/unraid/unraid-community-apps-starter>.

**Step 2 — add two files** (browser upload is fine — Add file → Upload
files):

- `ca_profile.xml` at the repository ROOT — copy from
  `deploy/unraid/ca_profile.xml`. This is the repository-level profile
  (who you are, what the repo contains, where support lives).
- `templates/openscrub-intel.xml` — copy from
  `deploy/unraid/openscrub-intel.xml`. One XML file per app; this one
  lists the Intel image with the `/dev/dri` device preconfigured.

**Step 3 — submit.** Sign in at <https://ca.unraid.net/submit> (an
Unraid forum account works), click **Start Submission**, and point it at
your `unraid-templates` repository. The portal live-scans the repo:
it parses the template XML, validates `ca_profile.xml`, checks for
duplicate listings, and previews the listing exactly as users will see
it. Fix anything it flags (edit the file on GitHub, then **Validate**
and **Scan** again), then submit. After moderator review it publishes
into the feed users browse from the Apps tab on their servers.

**Template notes:**

- Unraid exposes the port directly, so the template keeps OpenScrub's
  default self-signed HTTPS (`WebUI` is `https://[IP]:[PORT:8384]`) and
  the Overview warns about the one-time certificate prompt.
- The Intel GPU is passed as a **Device** config (`/dev/dri`) rather
  than an Extra Parameter, so users see it as a normal template field
  and can remove it on GPU-less hardware — OpenScrub degrades to CPU
  loudly and automatically.
- Appdata maps `/root/.local/share/OpenScrub` →
  `/mnt/user/appdata/openscrub` (jobs, reports, models, settings,
  certificates).
- A CUDA template can join the same repository later as
  `templates/openscrub-cuda.xml` (the README's Unraid/NVIDIA
  instructions already describe that setup manually); submit the Intel
  one first and add the second in place — the portal rescans your repo.
- The `<Date>` and `<Changes>` fields are yours to bump when you edit
  the template after publication.

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
