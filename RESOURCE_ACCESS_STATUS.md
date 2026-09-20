# scPLAD resource access status

Last verified: 2026-09-20 (Asia/Shanghai)

This file is the handoff record for checking the publication status of the
scPLAD code, data, checkpoints and interactive website. It intentionally
contains no credentials, access tokens, private server addresses or internal
filesystem paths.

## Current access summary

| Resource | Canonical URL | Current visibility | Anonymous access | Verified revision |
| --- | --- | --- | --- | --- |
| GitHub code and reproducibility repository | <https://github.com/zzgzanyr/scPLAD> | **Private** | No | `6c580e85b59f384e7f128705f3ac48a51b004593` |
| Hugging Face data and checkpoint repository | <https://huggingface.co/zhangzhigang/scPLAD> | **Private**, not gated | No (`401` from the public API) | `5625f1d4a0a7a57958426fe178a2b75fc4f3041f` |
| Hugging Face interactive Space | <https://huggingface.co/spaces/zhangzhigang/scplad-cellscape> | **Public** | Yes | `b2f36089b0922814aa630a568450dc377dd0b2b0` |
| Direct Space application | <https://zhangzhigang-scplad-cellscape.static.hf.space> | **Public** | Yes | Same Space revision |

At the time of this check, the only publicly accessible scPLAD resource is the
interactive Space. The GitHub repository and the full Hugging Face release are
not yet accessible to anonymous readers.

## GitHub repository

- Repository: `zzgzanyr/scPLAD`
- Default branch: `main`
- Visibility: `PRIVATE`
- Latest checked commit: `6c580e8`
- Latest commit subject: `Replace internal data paths with public release references`
- Scope: source code, portable configurations, baseline snapshots, compact
  results, figure-reproduction notebooks/scripts and documentation.
- Publication hygiene: author-machine paths, private compute addresses and
  author-specific server mount paths were removed from tracked files in commit
  `6c580e8`.
- Large datasets and selected checkpoints are not duplicated in GitHub. The
  README points to the Hugging Face release described below.

Important: making this repository public will expose all tracked source code,
historical configurations, compact results and figure-reproduction materials.
Review these materials against the manuscript-submission policy before changing
the visibility.

## Hugging Face data and checkpoints

- Repository ID: `zhangzhigang/scPLAD`
- Repository type: model repository used as the combined release store
- Visibility: `private=True`
- Gating: disabled (`gated=False`)
- File count: 62
- Total logical file size: 5,196,063,173 bytes (approximately 5.20 GB decimal)

### Included release groups

| Release path | Files | Contents |
| --- | ---: | --- |
| `datasets/k562_only/` | 5 | K562 train, validation and test matrices, README and checksums |
| `datasets/cross_cell_line/` | 9 | Cross-cell-line train, validation, test, test-control, control-context and PatchAE-training matrices plus checksums/documentation |
| `k562_only/` | 16 | K562 gene order, prior, configuration, PatchAE and manuscript checkpoint |
| `cross_cell_line/` | 17 | Cross-cell-line gene order, prior, configuration, PatchAE and 300k manuscript checkpoint |
| `docs/` | 2 | Experiment registry and source-provenance documentation |
| `scripts/` | 2 | Task-specific inference entry points |
| `src/` | 3 | Shared scPLAD implementation |

Anonymous API access currently returns HTTP `401`. Therefore, documentation may
link to this repository, but external users cannot download the release until
its visibility is changed to public. After publication, the intended download
command is:

```bash
hf download zhangzhigang/scPLAD --local-dir external/scPLAD-assets
```

## Hugging Face interactive Space

- Space ID: `zhangzhigang/scplad-cellscape`
- SDK: static HTML/JavaScript
- Visibility: public
- Anonymous API status: HTTP `200`
- File count: 67
- Total logical file size: 140,376,889 bytes (approximately 140 MB decimal)
- `browser-model/`: 11 files, 77,659,377 bytes
- `data/`: 29 files, 46,551,049 bytes
- `assets/`: 14 files, 3,037,979 bytes
- `cells/`: 4 files, 8,586,883 bytes

Because the Space is public, both the application and its tracked Space files
can be inspected by anyone. This is the main pre-submission exposure point at
present. The Space contains browser inference assets and compact demonstration
data, not the complete 5.20 GB training release.

## Recheck commands

The following commands do not print or require secret values for public checks.

### GitHub

```bash
gh repo view zzgzanyr/scPLAD \
  --json nameWithOwner,visibility,url,defaultBranchRef,updatedAt
```

Expected current result: `visibility` is `PRIVATE`.

### Hugging Face full release, anonymous check

```bash
curl -sS -o /dev/null -w '%{http_code}\n' \
  https://huggingface.co/api/models/zhangzhigang/scPLAD
```

Expected current result: `401`. After a successful public release this should
return `200`.

### Hugging Face Space, anonymous check

```bash
curl -sS -o /dev/null -w '%{http_code}\n' \
  https://huggingface.co/api/spaces/zhangzhigang/scplad-cellscape
```

Expected current result: `200`.

### Owner-authenticated Hugging Face check

Run only on a machine where the owner has already authenticated with the
Hugging Face CLI. Do not place the token in this document or command history.

```bash
python3 - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
model = api.model_info("zhangzhigang/scPLAD")
space = api.space_info("zhangzhigang/scplad-cellscape")
print("release_private:", model.private)
print("release_revision:", model.sha)
print("space_private:", space.private)
print("space_revision:", space.sha)
PY
```

## Final public-release checklist

1. Decide whether the Space should remain public before manuscript submission.
2. Review the GitHub diff and repository contents before changing GitHub from
   private to public.
3. Change `zhangzhigang/scPLAD` to public only when the data and checkpoint
   release is approved.
4. Repeat all anonymous checks in this file from a logged-out browser or a
   machine without cached credentials.
5. Verify the README links, `hf download` command and release checksums from a
   clean directory.
6. Record the final public revisions or DOI in this file and in the manuscript
   Data Availability and Code Availability statements.

## Interpretation for another reviewing model

Do not infer that a repository is public merely because its URL is present in a
README or because the owner can open it while logged in. Treat anonymous HTTP
status and the platform visibility field as authoritative. As of the timestamp
above, the intended release layout is complete, but GitHub and the 5.20 GB
Hugging Face release remain private.
