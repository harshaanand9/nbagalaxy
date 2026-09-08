# NBA Galaxy

A 3-D galaxy of **every NBA player-season since 2016-17**, sorted into 16 play-style
archetypes, scored with 26 skill badges, and compared with a supervised
per-player similarity model.

Everything spans all five positions. Point guards through centers are clustered in one
space, standardized against one league-wide pool, and badged against one set of peers.
No archetype or badge branches on a player's listed position; only the comparison model
uses position, and only to pick which learned weights apply and to soften cross-position
matches.

```text
players     3,212 player-seasons   (829 players, 2016-17 .. 2025-26, MP >= 15 and GP >= 26)
positions   PG 627 / SG 779 / SF 591 / PF 619 / C 596
archetypes  16, from Euclidean k-means++ on 77 features in 6 weighted blocks
badges      26, in 7 skill families
comparisons learned + personalized weights over 239 features in 34 blocks
```

## Two different models

The site runs **two** models, and they answer different questions. Keeping them
separate is deliberate.

| | Archetypes / galaxy | Player comparisons |
|---|---|---|
| Question | which of 16 play-style groups is this | who does this player-season most resemble |
| Space | 77 features, 6 weighted blocks | 239 features, 34 blocks, 63 subgroups |
| Weights | fixed per block | learned, then personalized per player |
| Source | `fullseasonfeatures_16_17_25_26.csv` | `similarity_model_dataset.csv` (BBall Index) |
| Code | `backend/app.py` | `backend/similarity_engine.py` |

The clustering pipeline below defines archetypes, medoids and the 3-D galaxy
layout. Similar-player links are drawn by the comparison model in
[Player comparisons](#player-comparisons), which is the algorithm described in
the paper *NBA Player-Season Similarity Algorithm*.

## The clustering pipeline

```text
season-standardize each feature against every player in that season
  -> clip at +/- 3.50 z-scores
  -> 6 blocks, weighted
  -> Euclidean k-means++ at k = 16
```

Blocks and weights:

| Block | Features | Weight |
|---|---|---|
| ThreePT | 18 | 15% |
| MidRange | 10 | 15% |
| RimPressure | 10 | 15% |
| Playmaking | 10 | 15% |
| Defense | 7 | 15% |
| Playtypes | 22 | 25% |

Each block is scaled by `sqrt(weight / feature_count)` per coordinate, so a block's
influence comes from its weight rather than from how many columns it happens to contain.

Standardizing against the whole league rather than a position group is deliberate. A
center who takes six threes a game is genuinely unusual and the model should see that;
comparing him only to other centers would hide the thing that makes him distinctive.

### Why k = 16

k was swept from 10 to 28 and scored on subsample stability (re-cluster on 80% samples,
measure AMI/ARI against the full fit).

| k | AMI | ARI | size balance | note |
|---|---|---|---|---|
| 12 | 0.795 | 0.714 | 2.59 | all non-shooting bigs collapse into one group |
| **16** | **0.726** | **0.575** | **3.08** | **4 distinct big types + an interior playmaking hub** |
| 18 | 0.703 | 0.523 | 3.06 | extra clusters only split wings into near-duplicates |

Stability falls monotonically with k, so the raw numbers always favour small k. Below 16
the model stops telling the truth about big men: a rim-running lob threat, a physical
paint scorer, and a floor-spacing five all become one cluster. Above 16 the extra
clusters describe nothing new. 16 is where four genuinely different kinds of big and the
interior playmaking hub survive while the guard structure stays intact.

### The 16 archetypes

| # | Archetype | n | Medoid |
|---|---|---|---|
| 1 | Primary Offensive Engine | 140 | Trae Young 2021-22 |
| 2 | Downhill Table-Setting Point | 218 | Jeff Teague 2018-19 |
| 3 | Pull-Up Shooting Combo Guard | 235 | Anfernee Simons 2024-25 |
| 4 | Point-of-Attack Connector Guard | 247 | Jose Alvarado 2024-25 |
| 5 | Isolation-Heavy Mid-Range Maestro | 144 | Devin Booker 2023-24 |
| 6 | Two-Level Movement Shooter | 219 | JJ Redick 2019-20 |
| 7 | 3PT-Reliant Sharpshooter | 280 | Jamison Battle 2024-25 |
| 8 | Limited-Playmaking Scoring Wing | 302 | Frank Jackson 2018-19 |
| 9 | Corner-Spacing 3-and-D Wing | 295 | Javonte Green 2024-25 |
| 10 | High-Efficiency Off-Ball Forward | 258 | Dorian Finney-Smith 2021-22 |
| 11 | Interior Playmaking Hub | 98 | Domantas Sabonis 2023-24 |
| 12 | Skilled Two-Way Scoring Big | 109 | Serge Ibaka 2018-19 |
| 13 | Floor-Spacing Stretch Big | 219 | Maxi Kleber 2019-20 |
| 14 | Conventional Two-Way Big | 216 | John Collins 2017-18 |
| 15 | Physical Paint Finisher | 111 | Jarrett Allen 2023-24 |
| 16 | Vertical Spacing Rim Protector | 121 | Rudy Gobert 2025-26 |

Archetypes are behavioural, not positional: cluster 13 mixes PFs and Cs, and two players
both listed at center can land in completely different clusters.

Names and full write-ups live in `EUCLIDEAN_KMEANS_CLUSTER_NAME_BY_NUMBER` and
`EUCLIDEAN_KMEANS_CLUSTER_DESCRIPTION_BY_NUMBER` in `backend/app.py`.

### Nobody is excluded

Every player-season that clears the minutes and games cut is clustered and badged.
Earlier builds held out LeBron James, Ben Simmons, and Scottie Barnes, and capped
Defensive Lock-Down for Harden / Lillard / Curry / Doncic. Those hacks existed because
the pool was guards only: three forwards were being measured against a peer group they
did not belong to, and four guards read as elite defenders because a guard-only pool
contained no rim protection to compare them against. A league-wide pool removes both
problems at the source, so `EUCLIDEAN_KMEANS_LOCKED_EXCLUDED_NAMES`,
`PERCENTILE_AND_BADGE_EXCLUDED_NAMES` and `DEFENSE_SCORE_CAPPED_NAMES` are all empty.

## Player comparisons

Comparisons are a **supervised, per-player** model -- a port of `sim.ipynb`, and the
algorithm written up in *NBA Player-Season Similarity Algorithm*. It lives in
`backend/similarity_engine.py` and is bit-identical to the notebook: every fitted
array and every query distance matches to 0.0.

It returns three ranked lists -- offense, defense and overall -- plus the attention
profile showing which skills drove the comparison.

```text
within-season z-score every raw feature
  -> per-subgroup PCA + whitening (domain ridge)
  -> learned weights from same-player adjacent-season pairs
  -> blend with this player's peer-relative distinctiveness
  -> adaptive hierarchical sharpening
  -> role gates on Paint Defense and Defensive Rebounding
  -> offense/defense identity balance
  -> symmetric pair-averaged distance + soft position penalty
  -> sim = 100 / (1 + (d / median d)^2)
```

**Weights are learned, not chosen.** A non-negative logistic regression is trained on
pairs of consecutive same-player seasons: skills that best re-identify a player earn
more weight. Six models are fit -- guard/wing/big x offense/defense.

**Then personalized.** Each player-season is scored against same-season, same-position
peers, and attention shifts toward what it is genuinely unusual at. Sharpening is
adaptive: when the evidence is spread across many areas the profile stays broad, so
versatile players are not collapsed into a single trait.

**Offense/defense balance is per player.** Built from the LEBRON family, it decides only
how much of the *overall* distance comes from each side of the ball -- not which skills
matter. Keyonte George 2025-26 reads 0.83/0.17; Walker Kessler 2024-25 reads 0.31/0.69.

Distances are pair-averaged, so `d(a,b) == d(b,a)` and a comparison is never controlled
by one side alone.

### Fit on everyone, compared against the roster

The model **fits** on the full BBall Index population (4,162 player-seasons, 2015-16 on),
because within-season z-scores, subgroup PCA, peer percentiles and the continuity
learner are all population statistics. Only the **candidate pool** is restricted to the
site's 3,206 matched player-seasons, so every comp returned is a player-season that
exists in the galaxy and can be clicked.

Six player-seasons are in the galaxy but absent from BBall Index (Frank Jackson 2018-19 /
2020-21 / 2021-22, Jevon Carter 2025-26, Johnny Davis 2022-23, Terrence Jones 2016-17).
They are archetyped and badged as normal; the comparison panel says explicitly that no
comps can be computed for them rather than showing an empty list.

### Reading the block breakdown

Each comp reports its most-alike and most-different skill areas. These rank by
`divergence` -- the block's contribution to the pair distance divided by its pair weight,
i.e. the weight-averaged squared distance *inside* that block. Contribution alone would
make any block the model barely weights look like a similarity, since a small weight
shrinks its contribution regardless of how different the two players actually are.
Divergence answers "how alike are these two here"; contribution answers "how much did
this block move the total". Both are computed; the UI ranks on divergence.

### Rebuilding

```bash
python3 scripts/build_similarity_dataset.py    # only after a BBall Index refresh
python3 scripts/precompute_similarity_v4.py    # ~90s, writes similarity_v4.json
python3 scripts/precompute_galaxy_assets.py    # picks up the new edges
```

`build_similarity_dataset.py` merges the BBall Index table and its impact side-loads
into `backend/data/similarity_model_dataset.csv`, keeping every row but only the columns
the engine reads. It is the one step that needs files outside this repo.

> The engine fits at import time (a couple of seconds) and is used **only** by the
> precompute scripts. `backend/app.py` never imports it -- the web app serves
> `similarity_v4.json` and never fits anything in a request path.

## Badges

26 badges in 7 families. Every badge score is a same-season percentile against every
player-season in the dataset.

| Family | Badges |
|---|---|
| three_pt (7) | Deep Range Bomber, Catch and Shoot Converter, Contested-3PT Maker, Pull-Up 3PT Machine, Volume 3PT Shooter, 3PT Sniper, **Inside-Out Threat** |
| midrange (2) | Volume Mid-Range Shooter, Mid-Range Assassin |
| interior (3) | **Rim Finisher**, **Paint Craftsman**, **Lob and Cut Finisher** |
| rim_pressure (5) | Volume Slasher, Efficient Driver, Free Throw Generator, Inside-The-Arc Scorer, Dunker |
| scoring (1) | Walking Bucket |
| playmaking (4) | Drive and Kicker, Assist Generator, Efficient Passer, **Screen Assist Machine** |
| defense (4) | Active Hands, Defensive Lock-Down, **Rim Protector**, **Perimeter Stopper** |

Bold entries are new in this build; they cover the parts of the game that forwards and
centers own and that a guards-only badge set never had to measure.

### Three rules make one league-wide pool fair

**1. Opportunity gates.** Every badge only opens for players who do that thing at volume.
A center never clears the pull-up-three gate and is therefore never measured on it; a
point guard never clears the shot-contest gate for Rim Protector. Gates are always
behavioural. `Inside-Out Threat` gates on *contesting shots* rather than on *attempting
shots at the rim*, because driving guards attempt plenty of shots at the rim — what
actually identifies an interior player is defending there.

**2. Local accuracy comparisons.** Where volume and efficiency both matter, accuracy is
ranked against players with similar volume rather than against the whole league. Rim
Protector is the clearest case: opponent FG% difference is compared only against players
who contest a similar share of shots, because rim contests are converted at a far higher
baseline rate than perimeter contests and an unadjusted number would punish the players
doing the most defensive work.

**3. Split defensive badges by location.** `Active Hands` dropped blocks and became the
perimeter-disruption badge (steals, deflections, charges drawn); blocks moved to
`Rim Protector`. `Perimeter Stopper` is Rim Protector's mirror, gated on low contest
volume *and* low opponent FGA. Without one league-wide pool collapsing every defensive
badge onto centers, guards and bigs each have a defensive badge they can actually win.

### Tiers are calibrated, not hand-picked

`BADGE_TIER_THRESHOLDS` in `backend/badge_engine.py` is solved, not chosen, so that every
badge lands on about the same share of the league:

```text
diamond  ~0.6% of the league        gold    ~2.4% (cumulative)
silver   ~6%   (cumulative)         bronze  ~12%  (cumulative)
```

Bronze means "great at this"; diamond means "the best in the league at this". About a
quarter of player-seasons earn no badge at all, which is intended: badges mark players
who are great to elite at a skill, not everyone who plays.

Re-solve after any data refresh, then paste the printed table into `badge_engine.py`:

```bash
python3 scripts/calibrate_badge_thresholds.py
```

## Data

The backend reads these, all checked in under `backend/data`:

```text
fullseasonfeatures_16_17_25_26.csv          main feature table (3,212 x 212)
fullseasonfeatures_player_comps_real.csv    D-LEBRON side-load
fullseasonfeatures_13_14_25_26_pullup.csv   pull-up 2PA/game side-load
euclidean_kmeans_locked_assignments.csv     the 16 archetype assignments
similarity_model_dataset.csv                comparison-model features (4,162 x 283)
similarity_v4.json                          precomputed comps + attention
```

Override the main table with `CLUSTER_DATASET_PATH`, and the assignments with
`EUCLIDEAN_KMEANS_LOCKED_ASSIGNMENTS_PATH`.

> The assignments file is the source of truth for archetypes. `cluster_raw` must equal
> `cluster - 1`, because the site derives the displayed cluster number as
> `cluster_raw + 1`. If those disagree, every archetype name attaches to the wrong group.

## Refreshing site assets

One command from the project root regenerates everything derived:

```bash
python3 scripts/precompute_all_site_assets.py
```

That writes:

```text
backend/data/similarity_v4.json
backend/data/galaxy_precomputed.json
backend/data/similar_players_precomputed_production.csv
backend/data/archetype_edges.csv
backend/data/cluster_medoids.csv
backend/data/archetype_labels.json
backend/data/player_badges.csv
backend/data/player_skill_breakdowns.json
backend/data/player_three_pt_breakdowns.json
backend/data/player_breakdown_manifest.json
frontend/public/precomputed/default_bootstrap.json
```

Then regenerate the per-player static assets, which read the full bootstrap:

```bash
python3 scripts/precompute_static_player_assets.py
```

Pass `--skip-headshots` and `--skip-badge-assets` to skip the two steps that need network
access or local source images.

Cluster labels, medoids, similar-player edges, badges, percentile payloads, detail panels
and cluster reports are all derived, so they must be regenerated whenever the source CSV
or the assignments change. The response cache keys on both the dataset mtime and the
assignments mtime, so a re-clustering invalidates it automatically.

## 3D galaxy

```text
X_truth   = season-standardized, clipped, block-weighted 77-D space
X_display = 3D UMAP when umap-learn is available, else 3D PCA
```

Constellation edges, medoids and archetype links are all selected in `X_truth`, and the
3D coordinates are only a display layer. **Similar-player links are not** -- they come
from the comparison model above.

Clicking a player draws lines to that player-season's nearest comps. `ARCHETYPES` draws
same-cluster edges from an MST plus a few same-cluster kNN links. `AUTO-SPIN` toggles the
idle camera orbit.

## Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Optional: `CLUSTER_CACHE_DIR` to relocate the response cache.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

### Static per-player assets

Opening a player reads flat files from the CDN instead of calling the backend:

```text
frontend/public/precomputed/players/<slug>.json          detail + skill + 3PT
frontend/public/precomputed/comps/<slug>.json            SIMILAR_PLAYERS
frontend/public/precomputed/cluster_reports/<n>.json
frontend/public/precomputed/comparison_options.json
```

`<slug>` is the player_key with runs of non-alphanumerics collapsed to `_`, so
`Nikola Jokic||2025-26||DEN||C` becomes `Nikola_Jokic_2025_26_DEN_C`. `App.jsx` derives it
with the same rule, so there is no index to keep in sync; the generator asserts the mapping
is collision-free (3212 keys, 3212 slugs) before it writes.

```bash
python3 scripts/precompute_static_player_assets.py
```

Every fetch still falls back to the API when a file is missing, so an unexported player
degrades to the old behaviour instead of breaking. Six player-seasons have no comps file
because the v4 model has no entry for them.

With these in place the site renders entirely from static files. The backend is only a
fallback, and nothing in the UI waits on it.

### The two bootstrap files

`scripts/precompute_default_bootstrap.py` writes the first-load payload twice:

| File | Size | In git | Used by |
| --- | --- | --- | --- |
| `default_bootstrap.json` | ~18 MB | yes | the deployed site |
| `default_bootstrap.full.json` | ~108 MB | no | local `npm run dev` |

Both draw the same galaxy -- same players, same archetypes. The full one additionally
bundles every player-detail panel and cluster report, so `npm run dev` needs no backend
running on `:8000`. That extra 90 MB is a click-time cache: `App.jsx` reads those two keys
with `?? {}` and falls back to the API when they are absent, which is what the deployed
site does. It has to, because the full file is past GitHub's 100 MB limit and Vercel
deploys from GitHub.

`App.jsx` picks between them on its own: `import.meta.env.DEV` is replaced at build time,
so `npm run dev` tries the full file first and a production build requests only the slim
one. Nothing to configure, and no wasted request in production. Setting
`VITE_DEFAULT_BOOTSTRAP_URL` still overrides both.

From a fresh clone the slim file is already there and the site renders; generate the full
one only if you want detail panels without running the backend.

## Badge icons

The badge strip renders each badge's source icon and converts it to a black silhouette in
CSS (`brightness(0)`). The seven badges added in this build ship **inline SVG data URIs**
in `frontend/src/badges.js` rather than remote URLs, so they have no external host that
can 404. The older badges still point at their original source URLs; if one of those is
wrong the icon fails visibly rather than silently drawing a replacement.

```text
bronze / silver / gold : flat coin + black silhouette
diamond                : wider light-blue gem + black silhouette
```

## Standalone weighted blocked-PCA experiment

`scripts/run_weighted_blocked_pca_kmeans.py` is a separate research runner against the
external BBall Index dataset. It is **not** wired to the site and writes to the ignored
`outputs/` directory.

```bash
python3 scripts/run_weighted_blocked_pca_kmeans.py
python3 -m unittest tests/test_weighted_blocked_pca.py
```
