# Guard Cluster Lab

This version wires the site to the current `fullseasonfeatures_16_17_25_26.csv` schema, keeps the locked **Euclidean + k++ means** preset, and adds a 3D galaxy visualization.

## Default dataset

The backend expects this file by default:

```bash
/Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_16_17_25_26.csv
```

Override it with:

```bash
export CLUSTER_DATASET_PATH="/absolute/path/to/fullseasonfeatures_16_17_25_26.csv"
```

## Locked Euclidean + k++ means preset

When the UI is set to:

```text
Algorithm: k++ means
Distance metric: Euclidean
```

then the backend forces:

```text
k = 12
features = backend locked Euclidean feature preset
cluster labels = backend/data/euclidean_kmeans_locked_assignments.csv
truth space = season-standardized/clipped -> blockwise PCA at 0.90 variance -> equal block weighting -> Euclidean distance
```

The frontend disables feature selection and the K slider for this mode. The assignment file preserves the original locked cluster IDs, while the UI displays cluster numbers as 1-12.

## 3D galaxy mode

The galaxy uses two spaces:

```text
X_truth   = blocked PCA 0.90 weighted Euclidean clustering space
X_display = 3D UMAP coordinates when umap-learn is available, otherwise 3D PCA fallback
```

The important behavior is that similar-player links, cluster constellation edges, medoids, and archetype links are selected from `X_truth`. The 3D coordinates are only the display layer.

Clicking a player draws constellation lines to that player-season's four nearest non-self player comps. Turning on `ARCHETYPES` draws sparse same-cluster constellation edges built from an MST plus small same-cluster kNN links.

## Optional precompute step

The backend can compute and cache the galaxy on first load, but production should precompute the static assets first:

```bash
python3 scripts/precompute_galaxy_assets.py \
  --dataset /Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_16_17_25_26.csv \
  --output-dir backend/data
```

This writes:

```text
backend/data/galaxy_precomputed.json
backend/data/similar_players_precomputed_production.csv
backend/data/archetype_edges.csv
backend/data/cluster_medoids.csv
backend/data/archetype_labels.json
```

The first-load frontend bootstrap can also be materialized as a static file:

```bash
python3 scripts/precompute_default_bootstrap.py \
  --dataset /Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_16_17_25_26.csv \
  --dlebron-dataset /Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_player_comps_real.csv
```

That writes `frontend/public/precomputed/default_bootstrap.json`, which contains the default config, galaxy payload, player detail panels, and cluster reports.

## Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Optional cache location:

```bash
export CLUSTER_CACHE_DIR="/absolute/path/to/cache_dir"
```

Optional locked-assignment override:

```bash
export EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH="/absolute/path/to/euclidean_kmeans_locked_assignments.csv"
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Then open the local URL printed by Vite.

## Player badges

This build adds a backend badge engine and frontend badge strip for the player report.

Backend entry points:

```bash
python3 scripts/precompute_player_badges.py \
  --dataset /Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_16_17_25_26.csv \
  --dlebron-dataset /Users/harsha/Desktop/PickPocketProjectOfficial/fullseasonfeatures_player_comps_real.csv \
  --output backend/data/player_badges.csv
```

The app also computes badges inside the normal `/api/player-details` flow, so the UI works even before the CSV is manually precomputed. Badge calculations are cached with the dataset load.

Badge assets:

The frontend badge strip now renders the exact source icon URLs directly in the UI and converts them to black silhouettes with CSS. The optional scripts are still available if you want to materialize local/generated assets manually:

```bash
npm run badges:bootstrap --prefix frontend
npm run badges:build --prefix frontend
```

The badge visual contract is now:

```text
bronze / silver / gold: flat coin shape + black silhouette from the exact source icon PNG
 diamond: wider light-blue gem/diamond shape + black silhouette from the exact source icon PNG
```

There are no generated fallback icon silhouettes in the frontend rendering path. If a source icon URL is wrong, the badge icon will fail visibly instead of silently drawing a fake replacement.

Updated source icons in this build:

```text
Inside-The-Arc Scorer: https://pngimg.com/uploads/number2/Number%202%20PNG%20images%20free%20download_PNG14925.png
Volume 3PT Shooter: https://www.shareicon.net/download/2015/10/05/651365_hand_512x512.png
```

The optional local materialization script still expects this local-only source if you want to build generated assets for the Volume Mid-Range Shooter badge:

```text
/Users/harsha/Downloads/pngegg.png
```

## 3D galaxy auto-spin

The 3D galaxy opens with a wider default camera angle and auto-spin enabled. The
`AUTO-SPIN` toggle next to the cluster color dots turns the slow camera orbit on/off.
Idle auto-spin is intentionally slow. After a player is selected, the selected player and
that player's five similar-player nodes stay fixed while the background galaxy points rotate
at an even slower ambient speed. Manual drag or wheel interaction temporarily pauses the
spin so the camera does not fight the user's cursor.

## One-command asset refresh

When `fullseasonfeatures_16_17_25_26.csv` changes, refresh all precomputed site assets with one command from the project root:

```bash
python3 scripts/precompute_all_site_assets.py
```

The defaults use the checked-in CSVs in `backend/data`. Pass `--dataset` and `--dlebron-dataset` only when refreshing from local source files outside the repo.

This runs the galaxy, badge, skill-breakdown, 3PT-breakdown, and frontend bootstrap precomputes in sequence. The main dataset still drives the existing skill-breakdown and badge features; `D-LEBRON` is side-loaded only where needed from the player-comps CSV. You still need to rerun the precompute when the source CSV changes because the cluster labels, medoids, similar-player edges, badges, percentile payloads, player detail panels, and cluster reports are derived files. The difference is that you no longer need to remember several separate script calls manually.

The new breakdown outputs are:

```text
backend/data/player_skill_breakdowns.json
backend/data/player_three_pt_breakdowns.json
backend/data/player_breakdown_manifest.json
frontend/public/precomputed/default_bootstrap.json
```

The backend checks those files first for `/api/player-skill-breakdown` and `/api/player-three-pt-breakdown`. If the manifest no longer matches the current dataset mtime or locked Euclidean feature signature, the backend falls back to the live calculation instead of serving stale values.
