import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import Plot from "react-plotly.js";
import { BADGE_META, BADGE_TIER_LABELS } from "./badges";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
// Tried in order. The full payload is gitignored (~108 MB) and bundles the player-detail
// and cluster-report caches, so `npm run dev` needs no backend on :8000. The slim payload
// is the committed one and is what deploys; it drops those two caches to clear GitHub's
// 100 MB limit, and the app fetches them from the API instead. `import.meta.env.DEV` is
// replaced at build time, so production never requests the file it does not have.
const DEFAULT_BOOTSTRAP_URLS = [
  import.meta.env.VITE_DEFAULT_BOOTSTRAP_URL,
  import.meta.env.DEV ? "/precomputed/default_bootstrap.full.json" : null,
  "/precomputed/default_bootstrap.json",
].filter(Boolean);
const SYNERGY_NOTE =
  "This stat uses Synergy defined offensive possessions instead of PBPStats derived offensive possessions.";

function buildClusterRequestKey(payload) {
  if (!payload) return "";
  const featuresKey = Array.isArray(payload.features) ? payload.features.join(",") : "";
  return [
    payload.algorithm,
    payload.distance_metric,
    payload.k,
    featuresKey,
  ].join("|");
}

function buildClusterReportRequestKey(payload) {
  if (!payload) return "";
  return [
    buildClusterRequestKey(payload),
    payload.cluster_number,
  ].join("|");
}

const CLUSTER_COLORS = [
  "#00D4E0",
  "#B77AFE",
  "#E01E37",
  "#55E6A5",
  "#F16F8B",
  "#66B5FF",
  "#F47FFF",
  "#7DF9FF",
  "#E7FF6B",
  "#FFAA5B",
  "#8FE388",
  "#FF8BD1",
];

const SKILL_BREAKDOWN_AXES = ["ThreePT", "MidRange", "RimPressure", "Playmaking", "Defense"];
const SKILL_BREAKDOWN_AXIS_LABELS = {
  Defense: "D-LEBRON",
};
const THREE_PT_BREAKDOWN_AXES = ["Off-Ball 3PT Shooting", "Self-Created 3PT Shooting", "3PT Volume"];


const PLAYER_COMPARISON_CATEGORY_OPTIONS = [
  { value: "traditional", label: "Traditional Stats" },
  { value: "three_pt", label: "3PT Stats" },
  { value: "midrange", label: "MidRange Stats" },
  { value: "rim_pressure", label: "Rim Pressure Stats" },
  { value: "playmaking", label: "Playmaking Stats" },
  { value: "defense", label: "Defensive Stats" },
];

const PLAYER_COMPARISON_MODE_OPTIONS = [
  { value: "raw_stats", label: "Raw Stats" },
  { value: "pace_adjusted_raw_stats", label: "Pace-Adjusted Raw Stats" },
  { value: "raw_frequencies", label: "Raw Frequencies" },
  { value: "raw_per_75", label: "Raw Per 75" },
  { value: "same_season_percentile", label: "Same-Season Percentile" },
  { value: "all_seasons_percentile", label: "All-Seasons Percentile" },
];

const ALGORITHM_LABELS = {
  kmeans: "K-Means++",
};

const DISTANCE_METRIC_LABELS = {
  euclidean: "Euclidean",
};

const DISTANCE_METRIC_OPTIONS = ["euclidean"];
const VISUALIZATION_MODE_OPTIONS = ["3d_galaxy"];
const VISUALIZATION_MODE_LABELS = {
  "2d_pca": "2D PCA plot",
  "3d_galaxy": "3D-Galaxy visualization",
};
const GALAXY_DEFAULT_CAMERA = {
  eye: { x: -0.92, y: 1.08, z: 0.54 },
  center: { x: 0, y: 0, z: 0 },
  up: { x: 0, y: 0, z: 1 },
};
const GALAXY_LAUNCH_CAMERA = {
  ...GALAXY_DEFAULT_CAMERA,
  eye: { x: -0.74, y: 0.87, z: 0.435 },
};
const GALAXY_CAMERA_INTERACTION_RESUME_MS = 900;
const GALAXY_CLICK_SELECTION_LOCK_MS = 360;
const GALAXY_MAX_VISIBLE_CLUSTER_CONSTELLATION_POINTS = 90;
const GALAXY_ARCHETYPE_OVERVIEW_MST_NODE_LIMIT = 45;
const GALAXY_MIN_CAMERA_DISTANCE = 0.10;
const GALAXY_MAX_CAMERA_DISTANCE = 4.2;
const GALAXY_FOCUS_MIN_CAMERA_DISTANCE = 0.20;
const GALAXY_FOCUS_MAX_CAMERA_DISTANCE = 0.92;
const GALAXY_SELECTED_NEIGHBOR_COUNT = 4;
const GALAXY_PLAYER_FOCUS_BASE_CAMERA = GALAXY_DEFAULT_CAMERA;
const GALAXY_PLAYER_FOCUS_ZOOM_DISTANCE = 0.32;
const GALAXY_PLAYER_FOCUS_MAX_ZOOM_DISTANCE = 0.80;
const GALAXY_PLAYER_FOCUS_VERTICAL_FOV_RADIANS = 1.10;
const GALAXY_PLAYER_FOCUS_VIEWPORT_PADDING = 0.78;
const GALAXY_PLAYER_FOCUS_SIDEBAR_BIAS_RATIO = 0.055;
const GALAXY_CLUSTER_FOCUS_MIN_CAMERA_DISTANCE = 0.42;
const GALAXY_CLUSTER_FOCUS_MAX_CAMERA_DISTANCE = 0.95;
const GALAXY_CLUSTER_FOCUS_RADIUS_MULTIPLIER = 1.45;
const GALAXY_FOCUS_CAMERA_ANIMATION_MS = 820;
const GALAXY_PLAYER_FOCUS_CAMERA_ANIMATION_MS = 1600;

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  const normalized = value.length === 3
    ? value.split("").map((char) => char + char).join("")
    : value;

  const int = Number.parseInt(normalized, 16);
  return {
    r: (int >> 16) & 255,
    g: (int >> 8) & 255,
    b: int & 255,
  };
}


function hexToRgba(hex, alpha = 1) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function blendClusterColor(memberships = []) {
  if (!memberships.length) return CLUSTER_COLORS[0];

  let r = 0;
  let g = 0;
  let b = 0;
  let totalWeight = 0;

  memberships.forEach((weight, index) => {
    const rgb = hexToRgb(CLUSTER_COLORS[index % CLUSTER_COLORS.length]);
    r += rgb.r * weight;
    g += rgb.g * weight;
    b += rgb.b * weight;
    totalWeight += weight;
  });

  if (totalWeight <= 0) return CLUSTER_COLORS[0];

  return `rgb(${Math.round(r / totalWeight)}, ${Math.round(g / totalWeight)}, ${Math.round(b / totalWeight)})`;
}

function getAlgorithmLabel(algorithm) {
  return ALGORITHM_LABELS[algorithm] ?? "K-Means++";
}

function getClusterControlLabel(algorithm) {
  return "K_CLUSTERS";
}

function getDistanceMetricLabel(metric) {
  return DISTANCE_METRIC_LABELS[metric] ?? metric ?? "Euclidean";
}

function getClusterColor(clusterNumber) {
  const parsedCluster = Number(clusterNumber);
  const normalizedCluster = Number.isFinite(parsedCluster) ? parsedCluster : 1;
  return CLUSTER_COLORS[(normalizedCluster - 1 + CLUSTER_COLORS.length * 100) % CLUSTER_COLORS.length];
}

function isNearWhiteColor(color) {
  const { r, g, b } = hexToRgb(color);
  return r >= 220 && g >= 232 && b >= 232;
}

function getSelectedPointColor(clusterNumber) {
  return isNearWhiteColor(getClusterColor(clusterNumber)) ? "#00D4E0" : "#FFFFFF";
}

function buildMembershipPieGradient(memberships = []) {
  if (!memberships.length) return getClusterColor(1);

  const total = memberships.reduce((sum, value) => sum + Math.max(Number(value) || 0, 0), 0);
  if (total <= 0) return getClusterColor(1);

  let angle = 0;
  const stops = memberships.map((value, index) => {
    const weight = Math.max(Number(value) || 0, 0) / total;
    const start = angle;
    angle += weight * 360;
    const end = index === memberships.length - 1 ? 360 : angle;
    return `${getClusterColor(index + 1)} ${start}deg ${end}deg`;
  });

  return `conic-gradient(${stops.join(", ")})`;
}

function normalizeSearchText(value = "") {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function formatPlayerSearchValue(point) {
  if (!point) return "";
  return `${point.player_name} · ${point.season} · ${point.position}`;
}

function formatShortSeasonLabel(season) {
  const seasonText = String(season ?? "").trim();
  const match = seasonText.match(/(\d{4})\s*[-–—]\s*(\d{2,4})/);
  if (!match) return seasonText;
  return `${match[1].slice(-2)}-${match[2].slice(-2)}`;
}

function formatPointSeasonName(point) {
  if (!point) return "";
  const shortSeason = formatShortSeasonLabel(point.season);
  return shortSeason ? `${shortSeason} ${point.player_name}` : String(point.player_name ?? "");
}

const SELECTOR_TOOLTIP_SHOW_DELAY_MS = 160;
const SELECTOR_TOOLTIP_HIDE_DELAY_MS = 70;
const SELECTOR_TOOLTIP_EXIT_MS = 220;
const GLOSSARY_OPEN_DELAY_MS = 55;
const GLOSSARY_EXIT_MS = 220;
const VIEW_GLITCH_SWAP_MS = 260;
const VIEW_GLITCH_TOTAL_MS = 620;

const PANEL_STORAGE_PREFIX = "cluster-site:v9";
const PANEL_LAYOUT_INIT_KEY = `${PANEL_STORAGE_PREFIX}:desktop-panels-initialized`;
const DEFAULT_LEFT_PANEL_WIDTH = 0;
const DEFAULT_RIGHT_PANEL_WIDTH = 300;
const DEFAULT_DRAWER_PANEL_WIDTH = 312;
const RESIZE_GUTTER_WIDTH = 12;
const MIN_LEFT_PANEL_WIDTH = 0;
const MAX_LEFT_PANEL_WIDTH = 420;
const MIN_RIGHT_PANEL_WIDTH = 0;
const MAX_RIGHT_PANEL_WIDTH = 420;
const MIN_CENTER_PANEL_WIDTH = 420;
const COLLAPSED_PANEL_THRESHOLD = 34;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

const PLOT_LAYOUT_MARGIN = { l: 38, r: 10, b: 36, t: 10 };
const PLOT_MIN_ZOOM = 1;
const PLOT_MAX_ZOOM = 12;
const PLOT_PINCH_ZOOM_SENSITIVITY = 0.006;
const PLOT_PAN_SENSITIVITY = 0.9;
const PLOT_PAN_SOFT_BOUND_RATIO = 0.58;
const PLOT_MAX_WHEEL_DELTA = 320;

function getPaddedRange(values) {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (!finiteValues.length) return [-1, 1];

  const minValue = Math.min(...finiteValues);
  const maxValue = Math.max(...finiteValues);
  const span = Math.max(maxValue - minValue, 0.001);
  const padding = Math.max(span * 0.075, 0.08);
  return [minValue - padding, maxValue + padding];
}

function getDefaultPlotAxisRange(points = []) {
  if (!points.length) return null;

  return {
    x: getPaddedRange(points.map((point) => Number(point.pc1))),
    y: getPaddedRange(points.map((point) => Number(point.pc2))),
  };
}

function clampAxisRangeToSoftBounds(range, defaultRange) {
  const span = Math.max(range[1] - range[0], 0.000001);
  const defaultSpan = Math.max(defaultRange[1] - defaultRange[0], 0.000001);
  const defaultCenter = (defaultRange[0] + defaultRange[1]) / 2;
  const currentCenter = (range[0] + range[1]) / 2;
  const zoomRatio = clamp(defaultSpan / span, PLOT_MIN_ZOOM, PLOT_MAX_ZOOM);
  const zoomProgress = clamp(1 - (1 / zoomRatio), 0, 1);
  const softSlack = defaultSpan * PLOT_PAN_SOFT_BOUND_RATIO * zoomProgress;
  const minCenter = defaultRange[0] - softSlack + span / 2;
  const maxCenter = defaultRange[1] + softSlack - span / 2;
  const nextCenter = minCenter <= maxCenter
    ? clamp(currentCenter, minCenter, maxCenter)
    : defaultCenter;

  return [nextCenter - span / 2, nextCenter + span / 2];
}

function clampPlotAxisRangeToSoftBounds(axisRange, defaultRange) {
  if (!axisRange || !defaultRange) return axisRange;

  return {
    x: clampAxisRangeToSoftBounds(axisRange.x, defaultRange.x),
    y: clampAxisRangeToSoftBounds(axisRange.y, defaultRange.y),
  };
}

function buildZoomedAxisRange(currentRange, defaultRange, fractionX, fractionYFromBottom, zoomFactor) {
  const currentXSpan = Math.max(currentRange.x[1] - currentRange.x[0], 0.000001);
  const currentYSpan = Math.max(currentRange.y[1] - currentRange.y[0], 0.000001);
  const defaultXSpan = Math.max(defaultRange.x[1] - defaultRange.x[0], 0.000001);
  const defaultYSpan = Math.max(defaultRange.y[1] - defaultRange.y[0], 0.000001);

  const nextXSpan = clamp(
    currentXSpan * zoomFactor,
    defaultXSpan / PLOT_MAX_ZOOM,
    defaultXSpan / PLOT_MIN_ZOOM
  );
  const nextYSpan = clamp(
    currentYSpan * zoomFactor,
    defaultYSpan / PLOT_MAX_ZOOM,
    defaultYSpan / PLOT_MIN_ZOOM
  );

  const anchorX = currentRange.x[0] + fractionX * currentXSpan;
  const anchorY = currentRange.y[0] + fractionYFromBottom * currentYSpan;
  const nextRange = {
    x: [
      anchorX - fractionX * nextXSpan,
      anchorX + (1 - fractionX) * nextXSpan,
    ],
    y: [
      anchorY - fractionYFromBottom * nextYSpan,
      anchorY + (1 - fractionYFromBottom) * nextYSpan,
    ],
  };

  return clampPlotAxisRangeToSoftBounds(nextRange, defaultRange);
}

function buildPannedAxisRange(currentRange, defaultRange, deltaX, deltaY, plotWidth, plotHeight) {
  const currentXSpan = Math.max(currentRange.x[1] - currentRange.x[0], 0.000001);
  const currentYSpan = Math.max(currentRange.y[1] - currentRange.y[0], 0.000001);
  const boundedDeltaX = clamp(deltaX, -PLOT_MAX_WHEEL_DELTA, PLOT_MAX_WHEEL_DELTA);
  const boundedDeltaY = clamp(deltaY, -PLOT_MAX_WHEEL_DELTA, PLOT_MAX_WHEEL_DELTA);
  const xShift = (boundedDeltaX / Math.max(plotWidth, 1)) * currentXSpan * PLOT_PAN_SENSITIVITY;
  const yShift = (-boundedDeltaY / Math.max(plotHeight, 1)) * currentYSpan * PLOT_PAN_SENSITIVITY;
  const nextRange = {
    x: [currentRange.x[0] + xShift, currentRange.x[1] + xShift],
    y: [currentRange.y[0] + yShift, currentRange.y[1] + yShift],
  };

  return clampPlotAxisRangeToSoftBounds(nextRange, defaultRange);
}

function getStoredNumber(key, fallback) {
  if (typeof window === "undefined") return fallback;
  const raw = window.localStorage.getItem(key);
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function getPanelBounds(layoutWidth, drawerOpen, galaxyFullscreenActive = false) {
  if (galaxyFullscreenActive) {
    const fullscreenMaxRight = Math.max(
      MIN_RIGHT_PANEL_WIDTH,
      Math.min(560, layoutWidth - MIN_CENTER_PANEL_WIDTH - RESIZE_GUTTER_WIDTH)
    );
    return { maxLeft: 0, maxRight: fullscreenMaxRight };
  }

  const drawerWidth = drawerOpen ? DEFAULT_DRAWER_PANEL_WIDTH : 0;
  const availableForPanels =
    layoutWidth - drawerWidth - RESIZE_GUTTER_WIDTH * 2 - MIN_CENTER_PANEL_WIDTH;

  const maxLeft = Math.max(
    MIN_LEFT_PANEL_WIDTH,
    Math.min(MAX_LEFT_PANEL_WIDTH, availableForPanels - MIN_RIGHT_PANEL_WIDTH)
  );
  const maxRight = Math.max(
    MIN_RIGHT_PANEL_WIDTH,
    Math.min(MAX_RIGHT_PANEL_WIDTH, availableForPanels - MIN_LEFT_PANEL_WIDTH)
  );

  return { maxLeft, maxRight };
}


const FEATURE_METADATA = {
  "Avg2ptShotDistance": {
    "label": "Avg2ptShotDistance",
    "description": "Average distance of a player's 2PT field-goal attempts.",
    "formula": "Source value from pbpstats shot-distance data."
  },
  "Avg3ptShotDistance": {
    "label": "Avg3ptShotDistance",
    "description": "Average distance of a player's 3PT field-goal attempts.",
    "formula": "Source value from pbpstats shot-distance data."
  },
  "OffPoss": {
    "label": "OffPoss",
    "description": "Player offensive possessions used as the denominator for offensive frequency and per-75 stats.",
    "formula": "Source value from possession data."
  },
  "DefPoss": {
    "label": "DefPoss",
    "description": "Player defensive possessions used as the denominator for defensive frequency and per-75 stats.",
    "formula": "Source value from possession data."
  },
  "crafted_cdpm": {
    "label": "Crafted CDPM",
    "description": "CraftedNBA defensive plus-minus style impact estimate.",
    "formula": "Source value from CraftedNBA advanced stats."
  },
  "D-LEBRON": {
    "label": "D-LEBRON",
    "description": "Defensive LEBRON impact estimate used as the defensive skill-breakdown source metric.",
    "formula": "Same-season league-wide percentile of the D-LEBRON column."
  },
  "crafted_box_creation": {
    "label": "Crafted box creation",
    "description": "CraftedNBA estimate of a player's offensive creation from box-score style signals.",
    "formula": "Source value from CraftedNBA advanced stats."
  },
  "crafted_passer_rating": {
    "label": "Crafted passer rating",
    "description": "CraftedNBA passing-impact rating used as a playmaking signal.",
    "formula": "Source value from CraftedNBA advanced stats."
  },
  "off_fouls_drawn_frequency": {
    "label": "Offensive fouls drawn frequency",
    "description": "Offensive fouls drawn per defensive possession.",
    "formula": "Offensive Fouls Drawn / DefPoss"
  },
  "zero_to_two_drib_3PA_Frequency": {
    "label": "0-2 dribble 3PA frequency",
    "description": "3PT attempts after zero, one, or two dribbles per offensive possession.",
    "formula": "(dribble_0_fg3a + dribble_1_fg3a + dribble_2_fg3a) / OffPoss"
  },
  "zero_to_two_drib_3FGA_Accuracy": {
    "label": "0-2 dribble 3PT accuracy",
    "description": "3PT accuracy on attempts after zero, one, or two dribbles.",
    "formula": "(dribble_0_fg3m + dribble_1_fg3m + dribble_2_fg3m) / (dribble_0_fg3a + dribble_1_fg3a + dribble_2_fg3a)"
  },
  "zero_to_two_drib_3FGA_avg_drib": {
    "label": "0-2 dribble 3PA average dribbles",
    "description": "Attempt-weighted average dribble count for 3PT attempts in the zero-to-two-dribble bucket.",
    "formula": "(1 * dribble_1_fg3a + 2 * dribble_2_fg3a) / (dribble_0_fg3a + dribble_1_fg3a + dribble_2_fg3a)"
  },
  "three_to_seven_plus_drib_3FGA_Frequency": {
    "label": "3+ dribble 3PA frequency",
    "description": "3PT attempts after three or more dribbles per offensive possession.",
    "formula": "(dribble_3_6_fg3a + dribble_7plus_fg3a) / OffPoss"
  },
  "three_to_seven_plus_drib_3FGA_Accuracy": {
    "label": "3+ dribble 3PT accuracy",
    "description": "3PT accuracy on attempts after three or more dribbles.",
    "formula": "(dribble_3_6_fg3m + dribble_7plus_fg3m) / (dribble_3_6_fg3a + dribble_7plus_fg3a)"
  },
  "pct_three_to_seven_plus_drib_3FGA_seven_plus": {
    "label": "Pct 3+ dribble 3PA from 7+ dribbles",
    "description": "Share of long-dribble 3PT attempts that come after seven or more dribbles.",
    "formula": "dribble_7plus_fg3a / (dribble_3_6_fg3a + dribble_7plus_fg3a)"
  },
  "zero_to_two_drib_2FGA_Frequency": {
    "label": "0-2 dribble 2PA frequency",
    "description": "2PT attempts after zero, one, or two dribbles per offensive possession.",
    "formula": "(dribble_0_fg2a + dribble_1_fg2a + dribble_2_fg2a) / OffPoss"
  },
  "zero_to_two_drib_2FGA_Accuracy": {
    "label": "0-2 dribble 2PT accuracy",
    "description": "2PT accuracy on attempts after zero, one, or two dribbles.",
    "formula": "(dribble_0_fg2m + dribble_1_fg2m + dribble_2_fg2m) / (dribble_0_fg2a + dribble_1_fg2a + dribble_2_fg2a)"
  },
  "zero_to_two_drib_2FGA_avg_drib": {
    "label": "0-2 dribble 2PA average dribbles",
    "description": "Attempt-weighted average dribble count for 2PT attempts in the zero-to-two-dribble bucket.",
    "formula": "(1 * dribble_1_fg2a + 2 * dribble_2_fg2a) / (dribble_0_fg2a + dribble_1_fg2a + dribble_2_fg2a)"
  },
  "three_to_seven_plus_drib_2FGA_Frequency": {
    "label": "3+ dribble 2PA frequency",
    "description": "2PT attempts after three or more dribbles per offensive possession.",
    "formula": "(dribble_3_6_fg2a + dribble_7plus_fg2a) / OffPoss"
  },
  "three_to_seven_plus_drib_2FGA_Accuracy": {
    "label": "3+ dribble 2PT accuracy",
    "description": "2PT accuracy on attempts after three or more dribbles.",
    "formula": "(dribble_3_6_fg2m + dribble_7plus_fg2m) / (dribble_3_6_fg2a + dribble_7plus_fg2a)"
  },
  "pct_three_to_seven_plus_drib_2FGA_seven_plus": {
    "label": "Pct 3+ dribble 2PA from 7+ dribbles",
    "description": "Share of long-dribble 2PT attempts that come after seven or more dribbles.",
    "formula": "dribble_7plus_fg2a / (dribble_3_6_fg2a + dribble_7plus_fg2a)"
  },
  "tight_very_tight_3fga_frequency": {
    "label": "Tight/very tight 3PA frequency",
    "description": "3PT attempts with a defender within four feet per offensive possession.",
    "formula": "(shot_contest_0_2_fg3a + shot_contest_2_4_fg3a) / OffPoss"
  },
  "tight_very_tight_3fga_accuracy": {
    "label": "Tight/very tight 3PT accuracy",
    "description": "3PT accuracy with a defender within four feet.",
    "formula": "(shot_contest_0_2_fg3m + shot_contest_2_4_fg3m) / (shot_contest_0_2_fg3a + shot_contest_2_4_fg3a)"
  },
  "open_3fga_frequency": {
    "label": "Open 3PA frequency",
    "description": "Open 3PT attempts per offensive possession.",
    "formula": "shot_contest_4_6_fg3a / OffPoss"
  },
  "open_3fga_accuracy": {
    "label": "Open 3PT accuracy",
    "description": "3PT accuracy on open attempts.",
    "formula": "shot_contest_4_6_fg3m / shot_contest_4_6_fg3a"
  },
  "Wide_Open_3FGA_Frequency": {
    "label": "Wide-open 3PA frequency",
    "description": "Wide-open 3PT attempts per offensive possession.",
    "formula": "shot_contest_6_plus_fg3a / OffPoss"
  },
  "pct_3fga_wide_open": {
    "label": "Pct 3PA wide open",
    "description": "Share of all 3PT attempts that were wide open.",
    "formula": "shot_contest_6_plus_fg3a / traditional_fg3a"
  },
  "Wide_Open_3FG_PCT": {
    "label": "Wide-open 3PT accuracy",
    "description": "3PT accuracy on wide-open attempts.",
    "formula": "shot_contest_6_plus_fg3m / shot_contest_6_plus_fg3a"
  },
  "tight_very_tight_2fga_frequency": {
    "label": "Tight/very tight 2PA frequency",
    "description": "2PT attempts with a defender within four feet per offensive possession.",
    "formula": "(shot_contest_0_2_fg2a + shot_contest_2_4_fg2a) / OffPoss"
  },
  "tight_very_tight_2fga_accuracy": {
    "label": "Tight/very tight 2PT accuracy",
    "description": "2PT accuracy with a defender within four feet.",
    "formula": "(shot_contest_0_2_fg2m + shot_contest_2_4_fg2m) / (shot_contest_0_2_fg2a + shot_contest_2_4_fg2a)"
  },
  "open_2fga_frequency": {
    "label": "Open 2PA frequency",
    "description": "Open and wide-open 2PT attempts per offensive possession.",
    "formula": "(shot_contest_4_6_fg2a + shot_contest_6_plus_fg2a) / OffPoss"
  },
  "open_2fga_accuracy": {
    "label": "Open 2PT accuracy",
    "description": "2PT accuracy on open and wide-open attempts.",
    "formula": "(shot_contest_4_6_fg2m + shot_contest_6_plus_fg2m) / (shot_contest_4_6_fg2a + shot_contest_6_plus_fg2a)"
  },
  "pull_up_3P_frequency": {
    "label": "Pull-up 3PA frequency",
    "description": "Pull-up 3PT attempts per offensive possession.",
    "formula": "pullup_pull_up_fg3a / OffPoss"
  },
  "pull_up_3P_accuracy": {
    "label": "Pull-up 3PT accuracy",
    "description": "3PT accuracy on pull-up attempts.",
    "formula": "pullup_pull_up_fg3m / pullup_pull_up_fg3a"
  },
  "pull_up_2P_frequency": {
    "label": "Pull-up 2PA frequency",
    "description": "Pull-up 2PT attempts per offensive possession.",
    "formula": "(pullup_pull_up_fga - pullup_pull_up_fg3a) / OffPoss"
  },
  "pull_up_2P_accuracy": {
    "label": "Pull-up 2PT accuracy",
    "description": "2PT accuracy on pull-up attempts.",
    "formula": "(pullup_pull_up_fgm - pullup_pull_up_fg3m) / (pullup_pull_up_fga - pullup_pull_up_fg3a)"
  },
  "catch_shoot_3P_frequency": {
    "label": "Catch-and-shoot 3PA frequency",
    "description": "Catch-and-shoot 3PT attempts per offensive possession.",
    "formula": "catch_shoot_catch_shoot_fg3a / OffPoss"
  },
  "catch_shoot_3P_accuracy": {
    "label": "Catch-and-shoot 3PT accuracy",
    "description": "3PT accuracy on catch-and-shoot attempts.",
    "formula": "catch_shoot_catch_shoot_fg3m / catch_shoot_catch_shoot_fg3a"
  },
  "MidRangeFrequency": {
    "label": "MidRangeFrequency",
    "description": "Midrange field-goal attempts per offensive possession.",
    "formula": "by_zone_statistics_mid_range_fga / OffPoss"
  },
  "MidRangeAccuracy": {
    "label": "MidRangeAccuracy",
    "description": "Midrange field-goal accuracy.",
    "formula": "by_zone_statistics_mid_range_fgm / by_zone_statistics_mid_range_fga"
  },
  "RestrictedArea_Frequency": {
    "label": "Restricted area frequency",
    "description": "Restricted-area attempts per offensive possession.",
    "formula": "by_zone_statistics_restricted_area_fga / OffPoss"
  },
  "RestrictedArea_Accuracy": {
    "label": "Restricted area accuracy",
    "description": "Restricted-area field-goal accuracy.",
    "formula": "by_zone_statistics_restricted_area_fgm / by_zone_statistics_restricted_area_fga"
  },
  "Paint_Non_RA_Frequency": {
    "label": "Paint non-RA frequency",
    "description": "Non-restricted-area paint attempts per offensive possession.",
    "formula": "by_zone_statistics_in_the_paint_non_ra_fga / OffPoss"
  },
  "Paint_Non_RA_Accuracy": {
    "label": "Paint non-RA accuracy",
    "description": "Non-restricted-area paint field-goal accuracy.",
    "formula": "by_zone_statistics_in_the_paint_non_ra_fgm / by_zone_statistics_in_the_paint_non_ra_fga"
  },
  "drive_fga_frequency": {
    "label": "Drive FGA frequency",
    "description": "Drive field-goal attempts per offensive possession.",
    "formula": "drives_drive_fga / OffPoss"
  },
  "drive_fg_pct": {
    "label": "Drive FG%",
    "description": "Field-goal accuracy on drives.",
    "formula": "drives_drive_fgm / drives_drive_fga"
  },
  "drive_fta_frequency": {
    "label": "Drive FTA frequency",
    "description": "Drive free-throw attempts per offensive possession.",
    "formula": "drives_drive_fta / OffPoss"
  },
  "Pct_Paint_FGA_from_drives": {
    "label": "Pct paint FGA from drives",
    "description": "Share of paint attempts that come from drives.",
    "formula": "drives_drive_fga / (by_zone_statistics_restricted_area_fga + by_zone_statistics_in_the_paint_non_ra_fga)"
  },
  "drib_tov_ratio": {
    "label": "Dribble-to-turnover ratio",
    "description": "Average dribbles per touch divided by turnovers per touch.",
    "formula": "touches_avg_drib_per_touch / (traditional_tov / touches_touches)"
  },
  "touch_frequency": {
    "label": "Touch frequency",
    "description": "Touches per offensive possession.",
    "formula": "touches_touches / OffPoss"
  },
  "potential_ast_tov_ratio": {
    "label": "Potential assist-to-turnover ratio",
    "description": "Potential assists per turnover.",
    "formula": "passing_potential_ast / traditional_tov"
  },
  "assist_frequency": {
    "label": "Assist frequency",
    "description": "Assists per offensive possession.",
    "formula": "passing_ast / OffPoss"
  },
  "pass_frequency": {
    "label": "Pass frequency",
    "description": "Passes made per offensive possession.",
    "formula": "passing_passes_made / OffPoss"
  },
  "Pass_shot_ratio": {
    "label": "Pass-shot ratio",
    "description": "Passes made per field-goal attempt.",
    "formula": "passing_passes_made / traditional_fga"
  },
  "potential_assist_frequency": {
    "label": "Potential assist frequency",
    "description": "Potential assists per offensive possession.",
    "formula": "passing_potential_ast / OffPoss"
  },
  "opp_players_fg_pct_difference": {
    "label": "Opponent FG% difference",
    "description": "Difference between opponent FG% against this defender and those opponents' average FG%.",
    "formula": "defense_dash_overall_pct_plusminus"
  },
  "contested_shot_frequency": {
    "label": "Contested shot frequency",
    "description": "Contested shots per defensive possession.",
    "formula": "hustle_contested_shots / DefPoss"
  },
  "Opp_players_fga_per_75_poss": {
    "label": "Opponent FGA per 75",
    "description": "Opponent shot attempts defended by the player per 75 defensive possessions.",
    "formula": "75 * defense_dash_overall_d_fga / DefPoss"
  },
  "dunks_per_75_poss": {
    "label": "Dunks per 75",
    "description": "Dunks per 75 offensive possessions.",
    "formula": "75 * dunks_per_game / OffPoss"
  },
  "Blocks_per_75": {
    "label": "Blocks per 75",
    "description": "Blocks per 75 defensive possessions.",
    "formula": "75 * traditional_blk / DefPoss"
  },
  "Steals_per_75": {
    "label": "Steals per 75",
    "description": "Steals per 75 defensive possessions.",
    "formula": "75 * traditional_stl / DefPoss"
  },
  "Deflections_per_75": {
    "label": "Deflections per 75",
    "description": "Deflections per 75 defensive possessions.",
    "formula": "75 * hustle_deflections / DefPoss"
  },
  "avg_drib_fg3a": {
    "label": "Average dribbles before 3PA",
    "description": "Attempt-weighted average dribble count before 3PT attempts.",
    "formula": "(1 * dribble_1_fg3a + 2 * dribble_2_fg3a + 4.5 * dribble_3_6_fg3a + 7.5 * dribble_7plus_fg3a) / (dribble_0_fg3a + dribble_1_fg3a + dribble_2_fg3a + dribble_3_6_fg3a + dribble_7plus_fg3a)"
  },
  "avg_drib_fg2a": {
    "label": "Average dribbles before 2PA",
    "description": "Attempt-weighted average dribble count before 2PT attempts.",
    "formula": "(1 * dribble_1_fg2a + 2 * dribble_2_fg2a + 4.5 * dribble_3_6_fg2a + 7.5 * dribble_7plus_fg2a) / (dribble_0_fg2a + dribble_1_fg2a + dribble_2_fg2a + dribble_3_6_fg2a + dribble_7plus_fg2a)"
  },
  "3P_Accuracy": {
    "label": "3P accuracy",
    "description": "Overall 3PT accuracy.",
    "formula": "traditional_fg3m / traditional_fg3a"
  },
  "3fga_frequency": {
    "label": "3PA frequency",
    "description": "3PT attempts per offensive possession.",
    "formula": "traditional_fg3a / OffPoss"
  },
  "pts_from_3s_per_75": {
    "label": "Points from 3s per 75",
    "description": "Scoring generated from made 3s per 75 offensive possessions.",
    "formula": "75 * (3 * traditional_fg3m) / OffPoss"
  },
  "pts_from_midrange_per_75": {
    "label": "Points from midrange per 75",
    "description": "Scoring generated from made midrange shots per 75 offensive possessions.",
    "formula": "75 * 2 * by_zone_statistics_mid_range_fgm / OffPoss"
  },
  "pts_from_drives_per_75": {
    "label": "Points from drives per 75",
    "description": "Estimated drive scoring per 75 offensive possessions.",
    "formula": "75 * ((2 * drive_fga_frequency * drive_fg_pct) + drive_fta_frequency)"
  },
  "avg_drib_per_touch": {
    "label": "Average dribbles per touch",
    "description": "Average dribbles per touch.",
    "formula": "touches_avg_drib_per_touch"
  },
  "fta_per_75": {
    "label": "FTA per 75",
    "description": "Free-throw attempts per 75 offensive possessions.",
    "formula": "75 * traditional_fta / OffPoss"
  },
  "drive_fta_per_75": {
    "label": "Drive FTA per 75",
    "description": "Drive free-throw attempts per 75 offensive possessions.",
    "formula": "75 * drives_drive_fta / OffPoss"
  },
  "pass_tov_ratio": {
    "label": "Pass-turnover ratio",
    "description": "Passes made per turnover.",
    "formula": "passing_passes_made / traditional_tov"
  },
  "pct_fga_3FGA": {
    "label": "Pct FGA from 3PA",
    "description": "Share of field-goal attempts that are 3PT attempts.",
    "formula": "traditional_fg3a / traditional_fga"
  },
  "pct_fga_MR": {
    "label": "Pct FGA from midrange",
    "description": "Share of field-goal attempts that are midrange attempts.",
    "formula": "by_zone_statistics_mid_range_fga / traditional_fga"
  },
  "pct_fga_drive_fga": {
    "label": "Pct FGA from drives",
    "description": "Share of field-goal attempts that come on drives.",
    "formula": "drives_drive_fga / traditional_fga"
  },
  "potential_assist_FGA_ratio": {
    "label": "Potential assist-FGA ratio",
    "description": "Potential assists per field-goal attempt.",
    "formula": "passing_potential_ast / traditional_fga"
  },
  "avg_closest_defender_3FGA": {
    "label": "Average closest defender on 3PA",
    "description": "Average closest-defender distance on 3PT attempts.",
    "formula": "Source value from closest-defender 3PT attempt data."
  },
  "avg_sec_per_touch": {
    "label": "Average seconds per touch",
    "description": "Average seconds the player holds the ball per touch.",
    "formula": "touches_avg_sec_per_touch"
  },
  "ASSISTS_ON_OFF": {
    "label": "Assists on/off",
    "description": "Team assist differential with the player on court versus off court.",
    "formula": "ASSISTS_on_court - ASSISTS_off_court"
  },
  "EFG_PCT_ON_OFF": {
    "label": "eFG% on/off",
    "description": "Team effective field-goal percentage differential with the player on court versus off court.",
    "formula": "EFG_PCT_on_court - EFG_PCT_off_court"
  },
  "PACE_ON_OFF": {
    "label": "Pace on/off",
    "description": "Team pace differential with the player on court versus off court.",
    "formula": "PACE_on_court - PACE_off_court"
  },
  "OPP_SHOT_QUALITY_ON_OFF": {
    "label": "Opponent shot quality on/off",
    "description": "Opponent shot-quality differential with the player on court versus off court.",
    "formula": "OPP_SHOT_QUALITY_on_court - OPP_SHOT_QUALITY_off_court"
  },
  "OPP_EFG_PCT_ON_OFF": {
    "label": "Opponent eFG% on/off",
    "description": "Opponent eFG% differential with the player on court versus off court.",
    "formula": "OPP_EFG_PCT_on_court - OPP_EFG_PCT_off_court"
  },
  "OPP_SHOT_QUALITY_TEAM": {
    "label": "Team opponent shot quality",
    "description": "Team-level opponent shot quality context.",
    "formula": "Source value from team opponent shot-quality data."
  },
  "OPP_EFG_PCT_TEAM": {
    "label": "Team opponent eFG%",
    "description": "Team-level opponent effective field-goal percentage context.",
    "formula": "Source value from team opponent eFG% data."
  },
  "pct_2p_fg_assisted": {
    "label": "Pct 2PT makes assisted",
    "description": "Share of made 2PT field goals that were assisted.",
    "formula": "assisted_2pm / (traditional_fgm - traditional_fg3m)"
  },
  "pct_3p_fg_assisted": {
    "label": "Pct 3PT makes assisted",
    "description": "Share of made 3PT field goals that were assisted.",
    "formula": "assisted_3pm / traditional_fg3m"
  }
};

const GLOSSARY_SECTIONS = [
  {
    "key": "context",
    "title": "CONTEXT",
    "features": [
      "OffPoss",
      "DefPoss",
      "Avg2ptShotDistance",
      "Avg3ptShotDistance"
    ]
  },
  {
    "key": "threept",
    "title": "THREE POINT",
    "features": [
      "tight_very_tight_3fga_frequency",
      "tight_very_tight_3fga_accuracy",
      "open_3fga_frequency",
      "open_3fga_accuracy",
      "Wide_Open_3FGA_Frequency",
      "Wide_Open_3FG_PCT",
      "pct_3fga_wide_open",
      "zero_to_two_drib_3PA_Frequency",
      "zero_to_two_drib_3FGA_Accuracy",
      "zero_to_two_drib_3FGA_avg_drib",
      "three_to_seven_plus_drib_3FGA_Frequency",
      "three_to_seven_plus_drib_3FGA_Accuracy",
      "pct_three_to_seven_plus_drib_3FGA_seven_plus",
      "pull_up_3P_frequency",
      "pull_up_3P_accuracy",
      "catch_shoot_3P_frequency",
      "catch_shoot_3P_accuracy",
      "avg_drib_fg3a",
      "3fga_frequency",
      "3P_Accuracy",
      "pts_from_3s_per_75",
      "pct_fga_3FGA",
      "pct_3p_fg_assisted",
      "avg_closest_defender_3FGA"
    ]
  },
  {
    "key": "midrange",
    "title": "MIDRANGE",
    "features": [
      "tight_very_tight_2fga_frequency",
      "tight_very_tight_2fga_accuracy",
      "open_2fga_frequency",
      "open_2fga_accuracy",
      "zero_to_two_drib_2FGA_Frequency",
      "zero_to_two_drib_2FGA_Accuracy",
      "zero_to_two_drib_2FGA_avg_drib",
      "three_to_seven_plus_drib_2FGA_Frequency",
      "three_to_seven_plus_drib_2FGA_Accuracy",
      "pct_three_to_seven_plus_drib_2FGA_seven_plus",
      "pull_up_2P_frequency",
      "pull_up_2P_accuracy",
      "avg_drib_fg2a",
      "MidRangeFrequency",
      "MidRangeAccuracy",
      "pct_fga_MR",
      "pts_from_midrange_per_75",
      "pct_2p_fg_assisted"
    ]
  },
  {
    "key": "rim",
    "title": "RIM PRESSURE",
    "features": [
      "RestrictedArea_Frequency",
      "RestrictedArea_Accuracy",
      "Paint_Non_RA_Frequency",
      "Paint_Non_RA_Accuracy",
      "drive_fga_frequency",
      "drive_fg_pct",
      "drive_fta_frequency",
      "Pct_Paint_FGA_from_drives",
      "dunks_per_75_poss",
      "pts_from_drives_per_75",
      "pct_fga_drive_fga",
      "fta_per_75",
      "drive_fta_per_75"
    ]
  },
  {
    "key": "playmaking",
    "title": "PLAYMAKING",
    "features": [
      "assist_frequency",
      "drib_tov_ratio",
      "avg_drib_per_touch",
      "potential_assist_frequency",
      "ASSISTS_ON_OFF",
      "assists_tov_ratio",
      "EFG_PCT_ON_OFF",
      "pts_created_from_assists",
      "PTS_PER_100_ON_OFF",
      "pts_created_to_tov_ratio",
      "THREE_PT_FG_PCT_ON_OFF",
      "crafted_box_creation",
      "crafted_passer_rating"
    ]
  },
  {
    "key": "defense",
    "title": "DEFENSE",
    "features": [
      "Blocks_per_75",
      "Steals_per_75",
      "Deflections_per_75",
      "off_fouls_drawn_frequency",
      "opp_players_fg_pct_difference",
      "contested_shot_frequency",
      "crafted_cdpm"
    ]
  },
  {
    "key": "onoff",
    "title": "ON/OFF + TEAM CONTEXT",
    "features": [
      "EFG_PCT_ON_OFF",
      "PACE_ON_OFF",
      "OPP_SHOT_QUALITY_ON_OFF",
      "OPP_EFG_PCT_ON_OFF",
      "OPP_SHOT_QUALITY_TEAM",
      "OPP_EFG_PCT_TEAM"
    ]
  }
];

function getFeatureMeta(feature) {
  return (
    FEATURE_METADATA[feature] ?? {
      label: feature,
      description: feature,
      formula: "",
      note: "",
    }
  );
}

function formatValue(value) {
  if (!Number.isFinite(value)) return "0.000";
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(3);
}

function formatPercentile(value) {
  if (!Number.isFinite(value)) return "0.0th";
  return `${value.toFixed(1)}th`;
}

function percentileTone(percentile) {
  if (percentile >= 85) return "tone-high";
  if (percentile >= 65) return "tone-good";
  if (percentile >= 35) return "tone-mid";
  if (percentile >= 15) return "tone-low";
  return "tone-bad";
}

function getGlossaryBadge(feature) {
  return feature.replace(/[^A-Za-z0-9]/g, "").slice(0, 8).toUpperCase();
}

const GLOSSARY_FEATURE_ORDER = GLOSSARY_SECTIONS.flatMap((section) => section.features);

function orderFeaturesByGlossary(features = []) {
  const seen = new Set();
  const output = [];

  GLOSSARY_FEATURE_ORDER.forEach((feature) => {
    if (features.includes(feature) && !seen.has(feature)) {
      output.push(feature);
      seen.add(feature);
    }
  });

  [...features]
    .filter((feature) => !seen.has(feature))
    .sort((a, b) => a.localeCompare(b))
    .forEach((feature) => {
      output.push(feature);
      seen.add(feature);
    });

  return output;
}

function formatSignedValue(value, digits = 2) {
  if (!Number.isFinite(value)) return '0.00';
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(digits)}`;
}

function getCenterViewLabel(view) {
  if (view === 'cluster_description') return 'cluster description';
  if (view === 'career_path') return 'career path';
  if (view === 'similar_players') return 'similar players';
  if (view === 'skill_breakdown') return 'skill breakdown';
  if (view === 'three_pt_breakdown') return '3PT breakdown';
  return 'plot';
}

function getVisualizationModeLabel(mode) {
  return VISUALIZATION_MODE_LABELS[mode] ?? VISUALIZATION_MODE_LABELS['2d_pca'];
}

function getSeasonStartYear(season) {
  const match = String(season ?? '').match(/(\d{4})/);
  return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
}

function sortSeasons(seasons = []) {
  return [...seasons].sort((a, b) => getSeasonStartYear(a) - getSeasonStartYear(b) || String(a).localeCompare(String(b)));
}

function getConfigClusterName(config, clusterNumber, algorithm = 'kmeans', distanceMetric = 'euclidean') {
  const key = String(clusterNumber);
  if (algorithm === 'kmeans' && distanceMetric === 'euclidean') {
    return config?.euclidean_kmeans_cluster_name_by_number?.[key]
      ?? config?.euclidean_kmeans_cluster_name_by_number?.[clusterNumber]
      ?? `Cluster ${clusterNumber}`;
  }

  return config?.cluster_name_by_number?.[key]
    ?? config?.cluster_name_by_number?.[clusterNumber]
    ?? `Cluster ${clusterNumber}`;
}

function formatSimilarityScore(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return "—";
  const normalizedValue = numericValue <= 1 ? numericValue * 100 : numericValue;
  return `${normalizedValue.toFixed(1)}`;
}

function formatBooleanLabel(value) {
  return value ? "YES" : "NO";
}

function normalizeComparableValue(value = "") {
  return String(value ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "").trim();
}

// Per-player payloads are precomputed to flat files under /precomputed and served
// from the CDN, so opening a player never waits on the backend -- no cold start, and
// no 404 from a backend whose dataset predates the player. Written by
// scripts/precompute_static_player_assets.py; this slug must match its slugify().
function playerAssetSlug(playerKey) {
  return String(playerKey ?? "").replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

// Resolves to null rather than throwing, so every caller can fall through to the API.
async function fetchStaticAsset(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function buildSimilarPlayersUrl({ sourcePoint, clusterData, config, activeClusterCount }) {
  const params = new URLSearchParams();
  params.set("player_name", sourcePoint.player_name);
  params.set("season", sourcePoint.season);
  params.set("pipeline", clusterData?.distance_metric ?? "euclidean");
  params.set("k", String(clusterData?.k ?? activeClusterCount));
  params.set("pca_variance_target", String(config?.pca_explained_var_target ?? ""));
  return `${API_BASE}/api/similar-players?${params.toString()}`;
}

function getHeatmapCellStyle(zValue) {
  const numericValue = Number(zValue) || 0;
  const clamped = clamp(numericValue, -3, 3);
  const magnitude = Math.min(Math.abs(clamped) / 3, 1);
  const lowColor = { r: 0, g: 212, b: 224 };
  const neutralColor = { r: 10, g: 21, b: 24 };
  const highColor = { r: 245, g: 111, b: 139 };
  const source = clamped >= 0 ? highColor : lowColor;
  const r = Math.round(neutralColor.r + (source.r - neutralColor.r) * magnitude);
  const g = Math.round(neutralColor.g + (source.g - neutralColor.g) * magnitude);
  const b = Math.round(neutralColor.b + (source.b - neutralColor.b) * magnitude);

  return {
    backgroundColor: `rgba(${r}, ${g}, ${b}, ${0.2 + 0.62 * magnitude})`,
    color: '#F4FBFC',
    borderColor: `rgba(${r}, ${g}, ${b}, ${0.58 + 0.24 * magnitude})`,
    boxShadow: magnitude > 0.72 ? `inset 0 0 0 1px rgba(${r}, ${g}, ${b}, 0.35)` : 'none',
  };
}

function ClusterFeatureStatCard({ item, summaryStatMode, valueMode }) {
  const meta = getFeatureMeta(item.feature);
  const displayValue = summaryStatMode === 'average'
    ? (valueMode === 'percentile' ? item.mean_percentile : item.mean_raw)
    : (valueMode === 'percentile' ? item.median_percentile : item.median_raw);

  const displayLabel = valueMode === 'percentile'
    ? formatPercentile(displayValue)
    : formatValue(displayValue);

  const robustTone = item.robust_z >= 0
    ? (item.robust_z >= 1.15 ? 'tone-high' : 'tone-good')
    : (item.robust_z <= -1.15 ? 'tone-bad' : 'tone-low');

  return (
    <div className='summary-card cluster-summary-card'>
      <div className='summary-card-name'>{meta.label}</div>
      <div className='summary-card-values cluster-summary-card-values'>
        <span>{displayLabel}</span>
        <div className='cluster-summary-chip-row'>
          <span className='cluster-summary-mini-chip'>{summaryStatMode === 'average' ? 'AVG' : 'MED'}</span>
          <span className={`pct-chip ${valueMode === 'percentile' ? percentileTone(displayValue) : robustTone}`}>
            {valueMode === 'percentile' ? displayLabel : formatValue(displayValue)}
          </span>
          <span className={`pct-chip ${robustTone}`}>RZ {formatSignedValue(item.robust_z, 2)}</span>
        </div>
      </div>
    </div>
  );
}

function ClusterDescriptionView({ report, clusterColor, onBack, loading, error }) {
  if (loading) {
    return <div className='cluster-description-empty-state cluster-description-loading-state'>HEAT_MAP_LOADING...</div>;
  }

  if (error) {
    return <div className='cluster-description-empty-state cluster-description-error-state'>{error}</div>;
  }

  if (!report) {
    return <div className='cluster-description-empty-state'>SELECT_A_CLUSTER_TO_VIEW_A_CLUSTER_DESCRIPTION.</div>;
  }

  const orderedFeatures = orderFeaturesByGlossary(report.feature_order ?? []);
  const orderedRows = report.heatmap_rows ?? [];
  const clusterTitle = report.cluster_title?.trim() || `Cluster ${report.cluster_number}`;

  return (
    <div className='cluster-description-layout'>
      <section className='cluster-description-copy neon-panel' style={{ '--cluster-color': clusterColor }}>
        <div className='cluster-description-title-bar'>
          <div className='cluster-description-title'>{clusterTitle}</div>
          {onBack && (
            <button type="button" className="career-path-back-btn cluster-description-back-btn" onClick={onBack}>
              BACK_TO_GALAXY
            </button>
          )}
        </div>
        <div className='cluster-description-body'>
          <div className='cluster-description-placeholder'>
            {report.description_text?.trim() || ''}
          </div>

          <div className='cluster-description-footer'>
            <div className='cluster-description-group'>
              <div className='report-subtitle report-subtitle-tight'>TYPICAL_PLAYERS</div>
              <div className='cluster-person-list'>
                {(report.typical_players ?? []).map((player) => (
                  <div key={`typical-${player.player_key}`} className='cluster-person-card'>
                    <div className='cluster-person-name'>{player.player_name}</div>
                    <div className='cluster-person-meta'>{player.season} · {player.teams_played} · {player.position}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className='cluster-description-group'>
              <div className='report-subtitle report-subtitle-tight'>NOTABLE_OUTLIERS</div>
              <div className='cluster-person-list'>
                {(report.notable_outliers ?? []).map((player) => (
                  <div key={`outlier-${player.player_key}`} className='cluster-person-card cluster-person-card-outlier'>
                    <div className='cluster-person-name'>{player.player_name}</div>
                    <div className='cluster-person-meta'>{player.season} · {player.teams_played} · {player.position}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className='cluster-description-heatmap neon-panel'>
        <div className='section-header'>STATISTICAL_FIGURES_PLOT</div>
        <div className='cluster-heatmap-shell'>
          <table className='cluster-heatmap-table'>
            <thead>
              <tr>
                <th className='cluster-heatmap-corner'>PLAYER</th>
                {orderedFeatures.map((feature) => (
                  <th key={`heatmap-head-${feature}`} className='cluster-heatmap-feature-header'>
                    {getFeatureMeta(feature).label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {orderedRows.map((row, rowIndex) => {
                const rowValueMap = new Map((row.values ?? []).map((value) => [value.feature, value]));
                return (
                  <tr key={`heatmap-row-${row.player_key ?? rowIndex}`} className={row.row_type === 'cluster_summary' ? 'cluster-heatmap-summary-row' : ''}>
                    <th className='cluster-heatmap-row-header'>
                      <div className='cluster-heatmap-player-name'>{row.row_type === 'cluster_summary' ? `Cluster ${report.cluster_number} Median` : row.player_name}</div>
                      {row.row_type !== 'cluster_summary' && (
                        <div className='cluster-heatmap-player-meta'>{row.season}</div>
                      )}
                    </th>
                    {orderedFeatures.map((feature) => {
                      const value = rowValueMap.get(feature) ?? { heatmap_z: 0 };
                      return (
                        <td
                          key={`heatmap-cell-${row.player_key ?? rowIndex}-${feature}`}
                          className='cluster-heatmap-cell'
                          style={getHeatmapCellStyle(value.heatmap_z)}
                        >
                          {formatSignedValue(value.heatmap_z, 1)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}



const FeatureTooltip = React.forwardRef(function FeatureTooltip(
  { feature, className = "", style = {}, visible = false },
  ref
) {
  const meta = getFeatureMeta(feature);
  if (!meta.description && !meta.formula && !meta.note) return null;

  return (
    <div
      ref={ref}
      className={`feature-tooltip ${visible ? "feature-tooltip-visible" : ""} ${className}`.trim()}
      style={style}
    >
      <div className="feature-tooltip-title">{meta.label}</div>
      {meta.description && <div className="feature-tooltip-description">{meta.description}</div>}
      {meta.formula && <div className="feature-tooltip-formula">Formula: {meta.formula}</div>}
      {meta.note && <div className="feature-tooltip-note">{meta.note}</div>}
    </div>
  );
});

function FeatureSelectorButton({
  feature,
  active,
  disabled = false,
  onClick,
  onHoverStart,
  onHoverMove,
  onHoverEnd,
}) {
  const meta = getFeatureMeta(feature);

  const handleHoverStart = (event) => {
    if (disabled) return;
    onHoverStart?.(feature, event.currentTarget.getBoundingClientRect());
  };

  const handleHoverMove = (event) => {
    if (disabled) return;
    onHoverMove?.(feature, event.currentTarget.getBoundingClientRect());
  };

  return (
    <button
      className={`feature-pill ${active ? "active" : ""} ${disabled ? "locked" : ""}`}
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={handleHoverStart}
      onMouseMove={handleHoverMove}
      onMouseLeave={onHoverEnd}
      onFocus={handleHoverStart}
      onBlur={onHoverEnd}
    >
      <span className={`feature-pill-dot ${active ? "active" : ""}`} aria-hidden="true" />
      <span className="feature-pill-label">{meta.label}</span>
    </button>
  );
}

function FeatureStatCard({ feature, value, percentile, label }) {
  const meta = getFeatureMeta(feature);
  const displayLabel = label || meta.label || feature;

  return (
    <div className="summary-card">
      <div className="summary-card-name">{displayLabel}</div>
      <div className="summary-card-values">
        <span>{formatValue(value)}</span>
        <span className={`pct-chip ${percentileTone(percentile)}`}>
          {formatPercentile(percentile)}
        </span>
      </div>
    </div>
  );
}

function ProbabilityBar({ memberships = [] }) {
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const hoverText = hoveredIndex == null
    ? ""
    : `Cluster ${hoveredIndex + 1} · ${(memberships[hoveredIndex] * 100).toFixed(1)}%`;

  return (
    <div className="probability-bar-shell">
      <div className="probability-bar-frame">
        <div className="probability-bar-track">
          {memberships.map((probability, index) => (
            <div
              key={`prob-${index + 1}`}
              className="probability-bar-segment"
              style={{
                width: `${Math.max(probability * 100, 0)}%`,
                background: CLUSTER_COLORS[index % CLUSTER_COLORS.length],
              }}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex((current) => (current === index ? null : current))}
              onFocus={() => setHoveredIndex(index)}
              onBlur={() => setHoveredIndex((current) => (current === index ? null : current))}
              tabIndex={0}
            />
          ))}
        </div>
        <div className={`probability-bar-readout ${hoverText ? "visible" : ""}`}>{hoverText}</div>
      </div>
    </div>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.8" />
      <path d="M16 16L21 21" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function ClusterListView({ segments = [], selectedPointKey, onSelectPlayer, onSelectCluster }) {
  if (!segments.length) {
    return <div className="cluster-list-empty">NO_CLUSTER_DATA</div>;
  }

  return (
    <div
      className="cluster-list-view"
      style={{ "--cluster-count": segments.length }}
    >
      {segments.map((segment) => (
        <section
          key={`cluster-list-${segment.cluster}`}
          className="cluster-list-segment"
          style={{ "--cluster-color": segment.color }}
        >
          <button
            type="button"
            className="cluster-list-segment-header cluster-list-segment-header-btn"
            onClick={() => onSelectCluster?.(segment.cluster)}
          >
            <span className="cluster-list-segment-title">Cluster {segment.cluster}</span>
            <span className="cluster-list-segment-count">{segment.count}</span>
          </button>

          <div className="cluster-list-segment-body">
            {segment.players.map((point) => (
              <button
                key={point.player_key}
                type="button"
                className={`cluster-list-player ${selectedPointKey === point.player_key ? "selected" : ""}`}
                onClick={() => onSelectPlayer?.(point)}
                title={`${point.player_name} · ${point.season}`}
              >
                <span className="cluster-list-player-name">{point.player_name}</span>
                <span className="cluster-list-player-season">{point.season}</span>
              </button>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function PlayerHeadshot({ src, name, size = "medium" }) {
  const resolvedSrc = src || "/headshots/fallback.svg";
  return (
    <div className={`player-headshot-frame player-headshot-${size}`}>
      <img
        className="player-headshot-img"
        src={resolvedSrc}
        alt={name ? `${name} headshot` : "Player headshot"}
        loading="lazy"
        onError={(event) => {
          event.currentTarget.onerror = null;
          event.currentTarget.src = "/headshots/fallback.svg";
        }}
      />
    </div>
  );
}

function BadgeTooltipPortal({ tooltip }) {
  if (!tooltip || typeof document === "undefined") return null;

  const className = `player-badge-tooltip-portal player-badge-tooltip-portal--${tooltip.placement}`;
  return createPortal(
    <div
      className={className}
      style={{
        left: `${tooltip.left}px`,
        top: `${tooltip.top}px`,
      }}
      role="tooltip"
    >
      <span className="player-badge-tooltip-tier">{tooltip.tierLabel}</span>
      <span className="player-badge-tooltip-name">{tooltip.badgeName}</span>
      {tooltip.rarityLabel && (
        <span className="player-badge-tooltip-rarity">{tooltip.rarityLabel}</span>
      )}
    </div>,
    document.body
  );
}

function PlayerBadges({ badges = [], compact = false, interactive = true }) {
  const [tooltip, setTooltip] = useState(null);
  const visibleBadges = Array.isArray(badges) ? badges.slice(0, compact ? 8 : 21) : [];
  if (!visibleBadges.length) return null;

  const showTooltip = (event, tierLabel, badgeName, rarityLabel = "") => {
    const rect = event.currentTarget.getBoundingClientRect();
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    const centerX = rect.left + rect.width / 2;
    const safeLeft = Math.min(Math.max(centerX, 132), Math.max(132, viewportWidth - 132));
    const canOpenAbove = rect.top >= 86;
    setTooltip({
      tierLabel,
      badgeName,
      rarityLabel,
      left: safeLeft,
      top: canOpenAbove ? rect.top - 10 : rect.bottom + 10,
      placement: canOpenAbove ? "above" : "below",
    });
  };

  const hideTooltip = () => setTooltip(null);

  return (
    <>
      <div className={`player-badges-row${compact ? " player-badges-row--compact" : ""}`} aria-label="Player badges">
        {visibleBadges.map((badge, index) => {
          const badgeId = badge.id || badge.badge_id;
          const tier = String(badge.tier || badge.badge_tier || "silver").toLowerCase();
          const meta = BADGE_META[badgeId] || {};
          const badgeName = badge.name || badge.badge_name || meta.name || "Player Badge";
          const tierLabel = BADGE_TIER_LABELS[tier] || tier.toUpperCase();
          const iconUrl = meta.iconUrl || "";
          const iconScale = Number(meta.iconScale ?? 0.6);
          const overlayGlyph = meta.overlayGlyph || "";
          const overlayClass = meta.overlayClass || "";
          const rarityLabel = badge.rarity_label || badge.rarityLabel || "";

          const BadgeTag = interactive ? "button" : "span";
          const badgeTagProps = interactive ? { type: "button", onFocus: (event) => showTooltip(event, tierLabel, badgeName, rarityLabel), onBlur: hideTooltip } : {};

          return (
            <BadgeTag
              key={`${badgeId}-${tier}-${index}`}
              className={`player-badge player-badge--${tier}`}
              style={{ "--badge-icon-scale": iconScale }}
              aria-label={`${tierLabel} ${badgeName}${rarityLabel ? ` · ${rarityLabel}` : ""}`}
              onMouseEnter={(event) => showTooltip(event, tierLabel, badgeName, rarityLabel)}
              onMouseLeave={hideTooltip}
              {...badgeTagProps}
            >
              <span className={`player-badge-core player-badge-core--${tier}`}>
                {iconUrl && (
                  <img
                    src={iconUrl}
                    alt=""
                    className="player-badge-source-icon"
                    loading="lazy"
                    draggable="false"
                  />
                )}
                {overlayGlyph && (
                  <span className={`player-badge-overlay player-badge-overlay--${overlayClass}`} aria-hidden="true">
                    {overlayGlyph}
                  </span>
                )}
              </span>
            </BadgeTag>
          );
        })}
      </div>
      <BadgeTooltipPortal tooltip={tooltip} />
    </>
  );
}


function SimilarPlayersView({
  data,
  sourcePoint,
  loading,
  error,
  onBack,
  onSelectSimilarPlayer,
}) {
  const sourceLabel = data
    ? `${data.player_name} · ${data.season}${data.team ? ` · ${data.team}` : ""}`
    : sourcePoint
      ? `${sourcePoint.player_name} · ${sourcePoint.season} · ${sourcePoint.teams_played}`
      : "NO_PLAYER_SELECTED";
  const [similarityDomain, setSimilarityDomain] = useState("overall");
  const [attentionOpen, setAttentionOpen] = useState(false);
  const [similarityMethodologyOpen, setSimilarityMethodologyOpen] = useState(false);
  const [blockedEuclideanOpen, setBlockedEuclideanOpen] = useState(false);

  // v4 responses carry three ranked lists; older ones only carry `similar_players`.
  const compsByDomain = data?.comps ?? null;
  const isV4 = Boolean(compsByDomain);
  const similarPlayers = isV4
    ? compsByDomain[similarityDomain] ?? []
    : data?.similar_players ?? [];
  const attention = data?.attention ?? null;
  const offWeightPct = attention ? Math.round(attention.off_weight * 100) : null;
  const defWeightPct = attention ? 100 - offWeightPct : null;

  return (
    <div className="similar-players-view">
      <div className="similar-players-header">
        <div>
          <div className="similar-players-title">SIMILAR_PLAYERS</div>
          <div className="similar-players-subtitle">SOURCE: {sourceLabel}</div>
          {data?.cluster_name && (
            <div className="similar-players-source-cluster">SOURCE_CLUSTER: {data.cluster_name}</div>
          )}
        </div>
        <div className="similar-players-header-actions">
          <button
            type="button"
            className="similarity-methodology-btn"
            onClick={() => setSimilarityMethodologyOpen(true)}
            aria-expanded={similarityMethodologyOpen}
          >
            Similarity Methodology
          </button>
          <button type="button" className="career-path-back-btn" onClick={onBack}>BACK_TO_GALAXY</button>
        </div>
      </div>

      {similarityMethodologyOpen && (
        <div
          className="similarity-methodology-page-backdrop"
          role="presentation"
          onClick={() => {
            setSimilarityMethodologyOpen(false);
            setBlockedEuclideanOpen(false);
          }}
          onMouseDown={(event) => event.stopPropagation()}
          onPointerDown={(event) => event.stopPropagation()}
        >
          <section
            className="similarity-methodology-page"
            role="dialog"
            aria-modal="true"
            aria-labelledby="similarity-methodology-title"
            onClick={(event) => event.stopPropagation()}
            onMouseDown={(event) => event.stopPropagation()}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <header className="similarity-methodology-page-header">
              <div>
                <div className="universe-modal-kicker">PLAYER SIMILARITY</div>
                <h2 id="similarity-methodology-title">Similarity Methodology</h2>
              </div>
              <button
                type="button"
                className="universe-modal-close-btn"
                onClick={() => {
                  setSimilarityMethodologyOpen(false);
                  setBlockedEuclideanOpen(false);
                }}
                aria-label="Close similarity methodology"
              >
                ×
              </button>
            </header>
            <div className="similarity-methodology-page-body">
              <p className="similarity-page-copy">
                Player comparisons come from a <span className="universe-cyan-text">supervised, per-player similarity model</span>. Unlike a fixed-weight tool, it learns which parts of the game re-identify a player across seasons, then re-weights that profile toward what makes <em>this</em> player-season distinctive. Every feature is standardized within its own season first, so era and pace never drive a match.
              </p>
              <UniverseAccordion
                title="How The Weights Are Chosen"
                open={blockedEuclideanOpen}
                onToggle={() => setBlockedEuclideanOpen((previousValue) => !previousValue)}
              >
                <p>
                  Features sit in a hierarchy of skill areas and subgroups. Each subgroup is rotated by PCA and whitened, so overlapping stats inside it cannot double-count and every subgroup starts with equal influence.
                </p>
                <p>
                  Weights are then <strong>learned</strong> by training on pairs of consecutive same-player seasons: the areas that best predict "this is the same player" earn more weight. Separate models are fit for guards, wings and bigs, and separately for offense and defense — six models in all.
                </p>
                <p>
                  Those league-wide weights are then <strong>personalized</strong>. Each player-season is scored against same-season, same-position peers to find what it is genuinely unusual at, and attention shifts toward those areas. Sharpening is adaptive: when a player's evidence is spread across many areas the profile deliberately stays broad, so versatile players are not collapsed into one trait.
                </p>
                <p>
                  Role gates keep low-opportunity defensive areas from inflating a match, and each player's offense/defense balance is set from impact metrics — which is why a defensive anchor is matched mostly on defense and a lead guard mostly on offense.
                </p>
                <p className="similarity-blocked-part-copy">
                  Distances are pair-averaged so the comparison reads the same in both directions, a soft penalty discourages guard-to-big matches without banning them, and scores are a monotone transform of distance: 100 / (1 + (d / median d)²).
                </p>
              </UniverseAccordion>
            </div>
          </section>
        </div>
      )}

      {isV4 && (
        <div className="similarity-v4-controls">
          <div className="similarity-domain-tabs" role="tablist" aria-label="Comparison domain">
            {[
              ["overall", "OVERALL"],
              ["offense", "OFFENSE"],
              ["defense", "DEFENSE"],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={similarityDomain === value}
                className={`similarity-domain-tab${similarityDomain === value ? " is-active" : ""}`}
                onClick={() => setSimilarityDomain(value)}
              >
                {label}
              </button>
            ))}
          </div>

          {attention && (
            <div className="similarity-identity-bar" title="How much of the overall comparison comes from each side of the ball">
              <span className="similarity-identity-label">IDENTITY</span>
              <div className="similarity-identity-track">
                <div className="similarity-identity-off" style={{ width: `${offWeightPct}%` }}>
                  OFF {offWeightPct}%
                </div>
                <div className="similarity-identity-def" style={{ width: `${defWeightPct}%` }}>
                  DEF {defWeightPct}%
                </div>
              </div>
              <button
                type="button"
                className="similarity-attention-btn"
                onClick={() => setAttentionOpen((previousValue) => !previousValue)}
                aria-expanded={attentionOpen}
              >
                {attentionOpen ? "HIDE_ATTENTION" : "SHOW_ATTENTION"}
              </button>
            </div>
          )}

          {attentionOpen && attention && (
            <div className="similarity-attention-panel">
              <p className="similarity-attention-copy">
                What the model weighted for this player-season, after learned continuity
                weights, their own distinctiveness, sharpening, role gates and the
                offense/defense balance. Shares sum to 100%.
              </p>
              <div className="similarity-attention-columns">
                <div className="similarity-attention-column">
                  <div className="similarity-attention-heading">SKILL_AREAS</div>
                  {attention.families.slice(0, 12).map((row) => (
                    <div key={`${row.family}-${row.domain}`} className="similarity-attention-row">
                      <span className={`similarity-attention-dot is-${String(row.domain).toLowerCase()}`} />
                      <span className="similarity-attention-name">{row.family}</span>
                      <span className="similarity-attention-bar">
                        <span style={{ width: `${Math.min(100, row.attention_pct * 8)}%` }} />
                      </span>
                      <strong>{row.attention_pct.toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
                <div className="similarity-attention-column">
                  <div className="similarity-attention-heading">INDIVIDUAL_SKILLSETS</div>
                  {attention.skillsets.slice(0, 12).map((row) => (
                    <div key={`${row.skillset}-${row.parent_block}-${row.domain}`} className="similarity-attention-row">
                      <span className={`similarity-attention-dot is-${String(row.domain).toLowerCase()}`} />
                      <span className="similarity-attention-name">{row.skillset}</span>
                      <span className="similarity-attention-bar">
                        <span style={{ width: `${Math.min(100, row.attention_pct * 8)}%` }} />
                      </span>
                      <strong>{row.attention_pct.toFixed(1)}%</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="similar-players-body">
        {loading && <div className="similar-players-empty">LOADING_SIMILAR_PLAYERS...</div>}
        {!loading && error && <div className="similar-players-empty similar-players-error">{error}</div>}
        {!loading && !error && !similarPlayers.length && (
          <div className="similar-players-empty">
            {data?.unavailable_reason
              ? `NO_COMPARISONS_AVAILABLE — ${data.unavailable_reason}`
              : "NO_SIMILAR_PLAYERS_FOUND"}
          </div>
        )}

        {!loading && !error && similarPlayers.length > 0 && (
          <div className="similar-players-grid">
            {similarPlayers.map((player) => (
              <button
                key={`${player.rank}-${player.player_name}-${player.season}-${player.team}`}
                type="button"
                className="similar-player-card"
                onClick={() => onSelectSimilarPlayer?.(player)}
              >
                <div className="similar-player-topline">
                  <span className="similar-rank">#{player.rank}</span>
                  <span className="similar-score">
                    SIM {formatSimilarityScore(
                      similarityDomain === "offense"
                        ? player.off_similarity
                        : similarityDomain === "defense"
                          ? player.def_similarity
                          : player.similarity_score,
                    )}
                  </span>
                </div>

                <div className="similar-player-identity-row">
                  <div className="similar-player-identity-copy">
                    <div className="similar-player-name">{player.player_name}</div>
                    <div className="similar-player-meta">{player.season} · {player.team || "—"} · {player.position || "—"}</div>
                    <div className="similar-player-cluster">{player.cluster_name || `Cluster ${player.cluster_number}`}</div>
                  </div>
                  <PlayerHeadshot src={player.headshot_url} name={player.player_name} size="small" />
                </div>

                <div className="similar-player-row">
                  <span>SAME_CLUSTER</span>
                  <strong>{formatBooleanLabel(player.same_cluster)}</strong>
                </div>

                <div className="similar-player-text-block">
                  <span>MAIN_SIMILARITIES</span>
                  <p>{player.strongest_similarity_blocks || "—"}</p>
                </div>

                <div className="similar-player-text-block">
                  <span>MAIN_DIFFERENCES</span>
                  <p>{player.biggest_difference_blocks || "—"}</p>
                </div>

                {isV4 ? (
                  <>
                    <div className="similar-player-section-label">DOMAIN_SIMILARITY</div>
                    <div className="similar-block-score-grid">
                      <div className={`similar-block-score${similarityDomain === "offense" ? " is-active" : ""}`}>
                        <span>OFFENSE</span>
                        <strong>{formatSimilarityScore(player.off_similarity)}</strong>
                      </div>
                      <div className={`similar-block-score${similarityDomain === "defense" ? " is-active" : ""}`}>
                        <span>DEFENSE</span>
                        <strong>{formatSimilarityScore(player.def_similarity)}</strong>
                      </div>
                      <div className={`similar-block-score${similarityDomain === "overall" ? " is-active" : ""}`}>
                        <span>OVERALL</span>
                        <strong>{formatSimilarityScore(player.overall_similarity)}</strong>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="similar-player-section-label">COMPONENT_LEVEL_SIMILARITIES</div>
                    <div className="similar-block-score-grid">
                      {Object.entries(player.block_scores ?? {}).map(([blockName, score]) => (
                        <div key={`${player.rank}-${blockName}`} className="similar-block-score">
                          <span>{blockName}</span>
                          <strong>{formatSimilarityScore(score?.similarity_score)}</strong>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function getSkillBreakdownAxisLabel(axis) {
  return SKILL_BREAKDOWN_AXIS_LABELS[axis] ?? axis;
}

function getRadarLabelLines(axis) {
  const displayAxis = getSkillBreakdownAxisLabel(axis);
  const labelOverrides = {
    "ThreePT": ["3PT"],
    "MidRange": ["Mid"],
    "RimPressure": ["Rim"],
    "Playmaking": ["Play"],
    "Playtypes": ["Types"],
    "D-LEBRON": ["D-LEBRON"],
    "Off-Ball 3PT Shooting": ["Off-Ball", "3PT"],
    "Self-Created 3PT Shooting": ["Self-Created", "3PT"],
    "3PT Volume": ["3PT", "Volume"],
  };
  if (labelOverrides[displayAxis]) return labelOverrides[displayAxis];

  const words = String(displayAxis).replace(/([a-z])([A-Z])/g, "$1 $2").split(/\s+/).filter(Boolean);
  if (words.length <= 1 || String(displayAxis).length <= 12) return [displayAxis];
  const midpoint = Math.ceil(words.length / 2);
  return [words.slice(0, midpoint).join(" "), words.slice(midpoint).join(" ")];
}

function PentagonSkillRadar({ title, eyebrow, scores = {}, color = CLUSTER_COLORS[0], axes = SKILL_BREAKDOWN_AXES }) {
  const width = 360;
  const height = 360;
  const center = width / 2;
  const axisCount = axes.length;
  const maxRadius = axisCount <= 3 ? 116 : axisCount > 5 ? 126 : 130;
  const labelRadius = axisCount <= 3 ? 154 : axisCount > 5 ? 164 : 162;

  const polarPoint = (axisIndex, radius) => {
    const angle = -Math.PI / 2 + (axisIndex * 2 * Math.PI) / axisCount;
    return {
      x: center + Math.cos(angle) * radius,
      y: center + Math.sin(angle) * radius,
    };
  };

  const polygonPoints = (radiusByAxis) => radiusByAxis
    .map((radius, axisIndex) => {
      const point = polarPoint(axisIndex, radius);
      return `${point.x},${point.y}`;
    })
    .join(" ");

  const values = axes.map((axis) => clamp(Number(scores?.[axis]) || 0, 0, 100));
  const profilePoints = polygonPoints(values.map((value) => (value / 100) * maxRadius));
  const safeTitle = String(title ?? "radar").replace(/[^a-z0-9]/gi, "-");

  return (
    <article className={`skill-radar-card ${axes.length > 5 ? "skill-radar-card-dense" : ""}`} style={{ "--skill-color": color, "--skill-color-soft": hexToRgba(color, 0.18) }}>
      <div className="skill-radar-card-header">
        <div className="skill-radar-eyebrow">{eyebrow}</div>
        <div className="skill-radar-title">{title}</div>
      </div>

      <div className="skill-radar-chart-wrap">
        <svg className="skill-radar-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} skill breakdown`}>
          <defs>
            <filter id={`skill-glow-${safeTitle}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {[20, 40, 60, 80, 100].map((level) => (
            <polygon
              key={`grid-${level}`}
              className="skill-radar-grid-ring"
              points={polygonPoints(Array(axisCount).fill((level / 100) * maxRadius))}
            />
          ))}

          {axes.map((axis, axisIndex) => {
            const endPoint = polarPoint(axisIndex, maxRadius);
            const labelPoint = polarPoint(axisIndex, labelRadius);
            const labelLines = getRadarLabelLines(axis);
            const textAnchor = Math.abs(labelPoint.x - center) < 10 ? "middle" : labelPoint.x > center ? "start" : "end";
            return (
              <g key={`axis-${axis}`}>
                <line className="skill-radar-axis-line" x1={center} y1={center} x2={endPoint.x} y2={endPoint.y} />
                <text
                  className="skill-radar-axis-label"
                  x={labelPoint.x}
                  y={labelPoint.y - ((labelLines.length - 1) * 6)}
                  textAnchor={textAnchor}
                  dominantBaseline="middle"
                >
                  {labelLines.map((line, lineIndex) => (
                    <tspan key={`${axis}-${lineIndex}`} x={labelPoint.x} dy={lineIndex === 0 ? 0 : 12}>{line}</tspan>
                  ))}
                </text>
              </g>
            );
          })}

          <polygon className="skill-radar-profile-fill" points={profilePoints} />
          <polygon className="skill-radar-profile-stroke" points={profilePoints} />
          {values.map((value, axisIndex) => {
            const dotPoint = polarPoint(axisIndex, (value / 100) * maxRadius);
            return <circle key={`skill-dot-${axisIndex}`} className="skill-radar-dot" cx={dotPoint.x} cy={dotPoint.y} r="3.8" />;
          })}
        </svg>
      </div>

      <div className="skill-score-list">
        {axes.map((axis) => (
          <div key={`${title}-${axis}`} className="skill-score-row">
            <span>{getSkillBreakdownAxisLabel(axis)}</span>
            <strong>{Math.round(clamp(Number(scores?.[axis]) || 0, 0, 100))}</strong>
          </div>
        ))}
      </div>
    </article>
  );
}

function getBreakdownPlayerDisplayLabel(data, sourcePoint, fallback = "NO_PLAYER_SELECTED") {
  const player = data?.player;
  const playerName = player?.player_name ?? player?.name ?? player?.label ?? sourcePoint?.player_name;
  const season = player?.season ?? sourcePoint?.season;
  if (!playerName) return fallback;

  const playerText = String(playerName).trim();
  const seasonText = season == null ? "" : String(season).trim();
  if (!seasonText) return playerText || fallback;
  if (playerText.startsWith(`(${seasonText})`) || playerText.startsWith(`${seasonText} `)) return playerText;
  return `(${seasonText}) ${playerText}`;
}

function buildBreakdownCards(data, sourcePoint, axes, baseColor = CLUSTER_COLORS[0]) {
  const playerSeason = data?.player?.season ?? sourcePoint?.season;
  return [
    {
      key: "player",
      eyebrow: "SELECTED_PLAYER",
      title: getBreakdownPlayerDisplayLabel(data, sourcePoint, "Selected Player"),
      scores: data?.player?.scores,
      color: baseColor,
    },
    {
      key: "cluster",
      eyebrow: "CLUSTER_MEDIAN",
      title: data?.cluster_median?.label ?? "Cluster Median",
      scores: data?.cluster_median?.scores,
      color: CLUSTER_COLORS[((data?.cluster_number ?? sourcePoint?.cluster ?? 1) - 1) % CLUSTER_COLORS.length],
    },
    {
      key: "guard",
      eyebrow: "SEASON_MEDIAN",
      title: data?.guard_median?.label ?? (playerSeason ? `Median Player ${playerSeason}` : "Median Player"),
      scores: data?.guard_median?.scores,
      color: "#DFF3F4",
    },
  ];
}

function MethodologyFeatureList({ items }) {
  return (
    <ul className="skill-methodology-feature-list">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

function SkillBreakdownMethodologyPage({ open, onClose }) {
  const [generalOpen, setGeneralOpen] = useState(true);
  const [threePtOpen, setThreePtOpen] = useState(false);
  const [deepRangeOpen, setDeepRangeOpen] = useState(false);
  const [catchShootOpen, setCatchShootOpen] = useState(false);
  const [contestedThreeOpen, setContestedThreeOpen] = useState(false);
  const [pullUpThreeOpen, setPullUpThreeOpen] = useState(false);
  const [threeVolumeOpen, setThreeVolumeOpen] = useState(false);
  const [threeAccuracyOpen, setThreeAccuracyOpen] = useState(false);
  const [midRangeOpen, setMidRangeOpen] = useState(false);
  const [midVolumeOpen, setMidVolumeOpen] = useState(false);
  const [midEfficiencyOpen, setMidEfficiencyOpen] = useState(false);
  const [rimPressureOpen, setRimPressureOpen] = useState(false);
  const [drivingVolumeOpen, setDrivingVolumeOpen] = useState(false);
  const [drivingEfficiencyOpen, setDrivingEfficiencyOpen] = useState(false);
  const [playmakingOpen, setPlaymakingOpen] = useState(false);
  const [passingVolumeOpen, setPassingVolumeOpen] = useState(false);
  const [passingEfficiencyOpen, setPassingEfficiencyOpen] = useState(false);
  const [craftedMetricsOpen, setCraftedMetricsOpen] = useState(false);
  const [defenseOpen, setDefenseOpen] = useState(false);
  const [excludedOpen, setExcludedOpen] = useState(false);

  if (!open) return null;

  return (
    <div
      className="skill-methodology-backdrop"
      role="presentation"
      onClick={onClose}
      onMouseDown={(event) => event.stopPropagation()}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <section
        className="skill-methodology-page"
        role="dialog"
        aria-modal="true"
        aria-labelledby="skill-methodology-title"
        onClick={(event) => event.stopPropagation()}
        onMouseDown={(event) => event.stopPropagation()}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header className="skill-methodology-header">
          <div>
            <div className="skill-methodology-kicker">SKILL BREAKDOWN</div>
            <h2 id="skill-methodology-title">Skill Breakdown Methodology</h2>
          </div>
          <button
            type="button"
            className="universe-modal-close-btn"
            onClick={onClose}
            aria-label="Close skill breakdown methodology"
          >
            ×
          </button>
        </header>

        <div className="skill-methodology-body">
          <UniverseAccordion
            title="General Rules"
            open={generalOpen}
            onToggle={() => setGeneralOpen((previousValue) => !previousValue)}
          >
            <p>
              When calculating the percentiles for both the overall block and the individual subsections, every percentile is calculated with respect to every NBA player from the same season, at every position.
            </p>
          </UniverseAccordion>

          <UniverseAccordion
            title="3PT Shooting Talent"
            open={threePtOpen}
            onToggle={() => setThreePtOpen((previousValue) => !previousValue)}
          >
            <UniverseAccordion title="Deep Range Shooting" open={deepRangeOpen} onToggle={() => setDeepRangeOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={[
                "Avg3ptShotDistance, where X is the player's Avg3ptShotDistance percentile",
                "3fga_frequency",
                "3P_Accuracy with respect to other players within the 100th percentile to X - 10 percentile range of Avg3ptShotDistance",
              ]} />
            </UniverseAccordion>

            <UniverseAccordion title="Catch-and-Shoot" open={catchShootOpen} onToggle={() => setCatchShootOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={[
                "catch_shoot_3P_frequency",
                "catch_shoot_3P_accuracy with respect to other players within the 100th percentile to X - 10 percentile range of the player's avg_closest_defender_3FGA, where X is the player's avg_closest_defender_3FGA percentile",
              ]} />
            </UniverseAccordion>

            <UniverseAccordion title="Contested 3PT Shot Making" open={contestedThreeOpen} onToggle={() => setContestedThreeOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={[
                "pct_3fga_wide_open, lower is better",
                "avg_closest_defender_3FGA, lower is better, where this percentile value is X",
                "3P_Accuracy with respect to other players within a 100th percentile to X - 10 percentile avg_closest_defender_3FGA range",
                "tight_very_tight_3fga_frequency",
                "3fga_frequency",
              ]} />
              <p className="skill-methodology-note">
                A player cannot be in the top 30 percentile for a season for Contested 3PT Shot Making if he does not have an above-average 3fga_frequency.
              </p>
            </UniverseAccordion>

            <UniverseAccordion title="Pull-Up 3PT Shooting" open={pullUpThreeOpen} onToggle={() => setPullUpThreeOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={[
                "pull_up_3P_frequency, where this percentile value is X",
                "pull_up_3P_accuracy with respect to other players within a 100th percentile to X - 10 percentile pull_up_3P_frequency range",
              ]} />
              <p className="skill-methodology-note">Percentiles are with respect to players in that specific season.</p>
            </UniverseAccordion>

            <UniverseAccordion title="3PT Volume" open={threeVolumeOpen} onToggle={() => setThreeVolumeOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={["3fga_frequency", "traditional_fg3a"]} />
            </UniverseAccordion>

            <UniverseAccordion title="3PT Accuracy" open={threeAccuracyOpen} onToggle={() => setThreeAccuracyOpen((previousValue) => !previousValue)} nested>
              <p>Weighted percentile of:</p>
              <MethodologyFeatureList items={[
                "3P_Accuracy, 70% weight, with respect to all players within 100th percentile to X - 10th percentile of avg_closest_defender_3FGA, where X is the player's avg_closest_defender_3FGA percentile",
                "3fga_frequency, 30% weight",
              ]} />
            </UniverseAccordion>
          </UniverseAccordion>

          <UniverseAccordion title="Mid-Range Shooting Talent" open={midRangeOpen} onToggle={() => setMidRangeOpen((previousValue) => !previousValue)}>
            <UniverseAccordion title="Volume Mid-Range Shooting" open={midVolumeOpen} onToggle={() => setMidVolumeOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={["MidRangeFrequency", "by_zone_statistics_mid_range_fga"]} />
            </UniverseAccordion>

            <UniverseAccordion title="Mid-Range Efficiency" open={midEfficiencyOpen} onToggle={() => setMidEfficiencyOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={[
                "MidRangeFrequency",
                "MidRangeAccuracy with respect to all players within the 100th to Y - 10 percentile range of tight_very_tight_2fga_frequency, where Y is the player's tight_very_tight_2fga_frequency percentile",
                "by_zone_statistics_mid_range_fga",
              ]} />
            </UniverseAccordion>
          </UniverseAccordion>

          <UniverseAccordion title="Rim-Pressure Talent" open={rimPressureOpen} onToggle={() => setRimPressureOpen((previousValue) => !previousValue)}>
            <UniverseAccordion title="Driving Volume" open={drivingVolumeOpen} onToggle={() => setDrivingVolumeOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={["drives_drive_fga", "drives_drives", "pts_from_drives_per_75"]} />
            </UniverseAccordion>

            <UniverseAccordion title="Driving Efficiency" open={drivingEfficiencyOpen} onToggle={() => setDrivingEfficiencyOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={[
                "drive_fg_pct with respect to all players within the drive_fga_frequency percentile range from the 100th percentile to the Y - 10 percentile, where Y is the player's drive_fga_frequency percentile",
                "pts_from_drives_per_75",
              ]} />
            </UniverseAccordion>
          </UniverseAccordion>

          <UniverseAccordion title="Playmaking Talent" open={playmakingOpen} onToggle={() => setPlaymakingOpen((previousValue) => !previousValue)}>
            <p className="skill-methodology-note">The overall Playmaking Talent score is the average of the subsection percentiles below.</p>
            <UniverseAccordion title="Passing Volume" open={passingVolumeOpen} onToggle={() => setPassingVolumeOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={["assist_frequency", "potential_assist_frequency", "pts_created_from_assists"]} />
            </UniverseAccordion>

            <UniverseAccordion title="Passing Efficiency" open={passingEfficiencyOpen} onToggle={() => setPassingEfficiencyOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={[
                "pts_created_from_assists",
                "assist_frequency",
                "potential_assist_frequency",
                "potential_assist_tov_ratio with respect to all players from the 100th percentile to the Y - 10th percentile of potential_assist_frequency, where Y is the player's potential_assist_frequency percentile",
                "assists_tov_ratio with respect to all players from the 100th percentile to the Z - 10th percentile of assist_frequency, where Z is the player's assist_frequency percentile",
                "pts_created_to_tov_ratio",
              ]} />
            </UniverseAccordion>

            <UniverseAccordion title="Crafted Metrics" open={craftedMetricsOpen} onToggle={() => setCraftedMetricsOpen((previousValue) => !previousValue)} nested>
              <p>Median percentile of:</p>
              <MethodologyFeatureList items={["crafted_box_creation", "crafted_passer_rating"]} />
            </UniverseAccordion>
          </UniverseAccordion>

          <UniverseAccordion title="D-LEBRON Skill Breakdown" open={defenseOpen} onToggle={() => setDefenseOpen((previousValue) => !previousValue)}>
            <p>
              The D-LEBRON Skill Breakdown percentile is the within-season league-wide percentile of the player's D-LEBRON rating.
            </p>
          </UniverseAccordion>

          <UniverseAccordion title="Excluded Players" open={excludedOpen} onToggle={() => setExcludedOpen((previousValue) => !previousValue)}>
            <p>
              LeBron James, Scottie Barnes, and Ben Simmons are excluded from all percentile calculations and badges.
            </p>
          </UniverseAccordion>
        </div>
      </section>
    </div>
  );
}

function SkillBreakdownView({ data, sourcePoint, loading, error, onBack, onOpenThreePtBreakdown }) {
  const axes = data?.axes?.length ? data.axes : SKILL_BREAKDOWN_AXES;
  const playerLabel = getBreakdownPlayerDisplayLabel(data, sourcePoint);
  const cards = buildBreakdownCards(data, sourcePoint, axes, CLUSTER_COLORS[0]);
  const [skillMethodologyOpen, setSkillMethodologyOpen] = useState(false);

  return (
    <div className="skill-breakdown-view">
      <div className="skill-breakdown-header">
        <div>
          <div className="section-header">// SKILL_BREAKDOWN</div>
          <div className="skill-breakdown-title">{playerLabel}</div>
          <div className="skill-breakdown-subtitle">
            Percentile-based five-part profile. Neutral shot-diet/touch-shape features are excluded from the score.
          </div>
          {data?.cluster_title && (
            <div className="skill-breakdown-cluster-line">ACTIVE_CLUSTER: #{data.cluster_number} · {data.cluster_title}</div>
          )}
        </div>
        <div className="skill-breakdown-header-actions">
          <button type="button" className="show-all-btn skill-breakdown-header-btn" onClick={() => setSkillMethodologyOpen(true)}>Skill Breakdown Methodology</button>
          <button type="button" className="show-all-btn skill-breakdown-header-btn" onClick={onOpenThreePtBreakdown}>3PT Breakdown</button>
          <button type="button" className="career-path-back-btn" onClick={onBack}>BACK_TO_GALAXY</button>
        </div>
      </div>

      <SkillBreakdownMethodologyPage
        open={skillMethodologyOpen}
        onClose={() => setSkillMethodologyOpen(false)}
      />

      {loading && <div className="skill-breakdown-empty">LOADING_SKILL_BREAKDOWN...</div>}
      {!loading && error && <div className="skill-breakdown-empty skill-breakdown-error">{error}</div>}
      {!loading && !error && !data && <div className="skill-breakdown-empty">NO_SKILL_BREAKDOWN_DATA</div>}

      {!loading && !error && data && (
        <div className="skill-breakdown-grid">
          {cards.map((card) => (
            <PentagonSkillRadar
              key={card.key}
              eyebrow={card.eyebrow}
              title={card.title}
              scores={card.scores}
              color={card.color}
              axes={axes}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ThreePtBreakdownView({ data, sourcePoint, loading, error, onBack }) {
  const axes = data?.axes?.length ? data.axes : THREE_PT_BREAKDOWN_AXES;
  const playerLabel = getBreakdownPlayerDisplayLabel(data, sourcePoint);
  const cards = buildBreakdownCards(data, sourcePoint, axes, "#E7FF6B");

  return (
    <div className="skill-breakdown-view three-pt-breakdown-view">
      <div className="skill-breakdown-header">
        <div>
          <div className="section-header">// 3PT_BREAKDOWN</div>
          <div className="skill-breakdown-title">{playerLabel}</div>
          <div className="skill-breakdown-subtitle">
            Three-part 3PT profile comparing off-ball shooting, self-created shooting, and 3PT volume.
          </div>
          {data?.cluster_title && (
            <div className="skill-breakdown-cluster-line">ACTIVE_CLUSTER: #{data.cluster_number} · {data.cluster_title}</div>
          )}
        </div>
        <button type="button" className="career-path-back-btn" onClick={onBack}>BACK_TO_GALAXY</button>
      </div>

      {loading && <div className="skill-breakdown-empty">LOADING_3PT_BREAKDOWN...</div>}
      {!loading && error && <div className="skill-breakdown-empty skill-breakdown-error">{error}</div>}
      {!loading && !error && !data && <div className="skill-breakdown-empty">NO_3PT_BREAKDOWN_DATA</div>}

      {!loading && !error && data && (
        <div className="skill-breakdown-grid three-pt-breakdown-grid">
          {cards.map((card) => (
            <PentagonSkillRadar
              key={card.key}
              eyebrow={card.eyebrow}
              title={card.title}
              scores={card.scores}
              color={card.color}
              axes={axes}
            />
          ))}
        </div>
      )}
    </div>
  );
}


function CareerPathView({
  playerName,
  timeline = [],
  clusterItems = [],
  selectedPointKey,
  selectedMissingSeason,
  onBack,
  onSelectQualifiedPoint,
  onSelectMissingSeason,
}) {
  const width = 1260;
  const height = 680;
  const margin = { top: 76, right: 46, bottom: 76, left: 374 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const clusterNumbers = clusterItems.map((item) => Number(item.cluster)).filter(Number.isFinite).sort((a, b) => a - b);
  const minCluster = clusterNumbers.length ? Math.min(...clusterNumbers) : 1;
  const maxCluster = clusterNumbers.length ? Math.max(...clusterNumbers) : 1;
  const clusterSpan = Math.max(maxCluster - minCluster, 1);
  const denomX = Math.max(timeline.length - 1, 1);

  const xForIndex = (index) => margin.left + (index / denomX) * plotWidth;
  const yForClusterValue = (clusterValue) => {
    const numericCluster = Number(clusterValue);
    const safeCluster = Number.isFinite(numericCluster) ? numericCluster : minCluster;
    const normalized = (safeCluster - minCluster) / clusterSpan;
    return margin.top + (1 - normalized) * plotHeight;
  };

  const interpolatedClusterForIndex = (index) => {
    const current = timeline[index];
    if (!current) return minCluster;
    if (current.qualified) return current.cluster;

    let previousIndex = -1;
    let nextIndex = -1;
    for (let i = index - 1; i >= 0; i -= 1) {
      if (timeline[i]?.qualified) {
        previousIndex = i;
        break;
      }
    }
    for (let i = index + 1; i < timeline.length; i += 1) {
      if (timeline[i]?.qualified) {
        nextIndex = i;
        break;
      }
    }

    if (previousIndex >= 0 && nextIndex >= 0) {
      const previousCluster = Number(timeline[previousIndex].cluster);
      const nextCluster = Number(timeline[nextIndex].cluster);
      const t = (index - previousIndex) / Math.max(nextIndex - previousIndex, 1);
      return previousCluster + (nextCluster - previousCluster) * t;
    }

    if (previousIndex >= 0) return timeline[previousIndex].cluster;
    if (nextIndex >= 0) return timeline[nextIndex].cluster;
    return minCluster;
  };

  const pathNodes = timeline.map((entry, index) => ({
    ...entry,
    x: xForIndex(index),
    y: yForClusterValue(interpolatedClusterForIndex(index)),
  }));
  const polylinePoints = pathNodes.map((node) => `${node.x},${node.y}`).join(' ');

  return (
    <div className="career-path-view">
      <div className="career-path-header">
        <div>
          <div className="section-header">// CAREER_PATH</div>
          <div className="career-path-title">{playerName || 'SELECTED PLAYER'}</div>
          <div className="career-path-subtitle">Cluster movement across qualified player-seasons in the active galaxy view.</div>
        </div>
        <button type="button" className="career-path-back-btn" onClick={onBack}>BACK_TO_GALAXY</button>
      </div>

      {!timeline.length ? (
        <div className="cluster-list-empty">NO_CAREER_PATH_DATA</div>
      ) : (
        <div className="career-path-chart-shell">
          <svg className="career-path-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${playerName} career path by cluster`}>
            <defs>
              <pattern id="career-dnq-stripes" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="8" height="8" fill="#f5f7f8" />
                <rect width="3" height="8" fill="#9aa4aa" opacity="0.62" />
              </pattern>
              <filter id="career-dot-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            <rect x="0" y="0" width={width} height={height} className="career-path-bg" />

            {clusterItems.map((item) => {
              const y = yForClusterValue(item.cluster);
              return (
                <g key={`career-y-${item.cluster}`} className="career-y-row">
                  <line x1={margin.left} x2={width - margin.right} y1={y} y2={y} className="career-grid-line" />
                  <circle cx={margin.left - 28} cy={y} r="7" fill={item.color} className="career-axis-cluster-dot" />
                  <text x={margin.left - 42} y={y + 4} textAnchor="end" className="career-axis-label">
                    #{item.cluster} {item.name}
                  </text>
                </g>
              );
            })}

            {timeline.map((entry, index) => {
              const x = xForIndex(index);
              return (
                <g key={`career-x-${entry.season}`} className="career-x-tick">
                  <line x1={x} x2={x} y1={margin.top} y2={height - margin.bottom} className="career-grid-line career-grid-line-vertical" />
                  <text x={x} y={height - 34} textAnchor="middle" className="career-axis-label career-season-label">{entry.season}</text>
                </g>
              );
            })}

            <line x1={margin.left} x2={width - margin.right} y1={height - margin.bottom} y2={height - margin.bottom} className="career-axis-line" />
            <line x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} className="career-axis-line" />
            <text x={margin.left + plotWidth / 2} y={height - 8} textAnchor="middle" className="career-axis-title">Seasons</text>
            <text x="26" y={margin.top + plotHeight / 2} textAnchor="middle" className="career-axis-title career-y-title" transform={`rotate(-90 26 ${margin.top + plotHeight / 2})`}>Cluster</text>

            {pathNodes.length > 1 && <polyline points={polylinePoints} className="career-path-line" />}

            {pathNodes.map((node) => {
              if (node.qualified) {
                const isSelected = selectedPointKey === node.point?.player_key;
                return (
                  <g key={`career-node-${node.season}`} className={`career-node ${isSelected ? 'selected' : ''}`}>
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={isSelected ? 13 : 10}
                      fill={getClusterColor(node.cluster)}
                      className="career-qualified-dot"
                      filter="url(#career-dot-glow)"
                      onClick={() => onSelectQualifiedPoint?.(node.point)}
                    />
                    <title>{`${node.season} · ${node.cluster_name}`}</title>
                  </g>
                );
              }

              const isSelected = selectedMissingSeason === node.season;
              return (
                <g key={`career-node-${node.season}`} className={`career-node career-node-dnq ${isSelected ? 'selected' : ''}`} onClick={() => onSelectMissingSeason?.(node)}>
                  <circle cx={node.x} cy={node.y} r={isSelected ? 12 : 10} fill="url(#career-dnq-stripes)" className="career-dnq-dot" />
                  <text x={node.x} y={node.y - 18} textAnchor="middle" className="career-dnq-label">Did Not Qualify</text>
                  <title>{`${node.season} · Did Not Qualify`}</title>
                </g>
              );
            })}
          </svg>
        </div>
      )}
    </div>
  );
}

function FeatureLockOverlay() {
  return (
    <div className="feature-lock-overlay" aria-hidden="true">
      <svg className="feature-lock-icon" viewBox="0 0 96 96" fill="none">
        <path
          d="M30 40V29C30 19.6112 37.6112 12 47 12H49C58.3888 12 66 19.6112 66 29V40"
          stroke="currentColor"
          strokeWidth="6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <rect
          x="19"
          y="40"
          width="58"
          height="44"
          rx="10"
          stroke="currentColor"
          strokeWidth="6"
        />
        <path
          d="M48 56V67"
          stroke="currentColor"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <circle cx="48" cy="54" r="5" fill="currentColor" />
      </svg>
    </div>
  );
}

function SearchClusterBadge({ point }) {
  if (!point) return null;

  return (
    <span
      className="search-cluster-badge search-cluster-badge-solid"
      style={{ "--cluster-color": getClusterColor(point.cluster) }}
      title={`Cluster ${point.cluster}`}
      aria-hidden="true"
    >
      <span>{point.cluster}</span>
    </span>
  );
}

function GlossarySection({ section, open, onToggle }) {
  return (
    <div className="glossary-section">
      <button className="glossary-section-header" onClick={onToggle}>
        <span className={`glossary-section-chevron ${open ? "open" : ""}`}>›</span>
        <span>{section.title}</span>
        <span className="glossary-section-count">({section.features.length})</span>
      </button>

      <div className={`glossary-section-body ${open ? "open" : ""}`}>
        <div className="glossary-section-inner">
          {section.features.map((feature) => {
            const meta = getFeatureMeta(feature);
            return (
              <div key={feature} className="glossary-card">
                <div className="glossary-card-header">
                  <span className="glossary-card-badge">{getGlossaryBadge(feature)}</span>
                  <div className="glossary-card-title-wrap">
                    <div className="glossary-card-title">{meta.label}</div>
                    <div className="glossary-card-key">{feature}</div>
                  </div>
                </div>
                {meta.description && <p className="glossary-card-copy">{meta.description}</p>}
                {meta.formula && (
                  <p className="glossary-card-formula">Formula: {meta.formula}</p>
                )}
                {meta.note && <p className="glossary-card-note">{meta.note}</p>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}


function cross(o, a, b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function buildConvexHull(points) {
  if (!points || points.length <= 3) return points ? [...points] : [];

  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }

  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const point = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }

  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function polygonCentroid(points) {
  if (!points.length) return { x: 0, y: 0 };
  const sum = points.reduce((acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }), { x: 0, y: 0 });
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function expandPolygon(points, scale = 1.06) {
  if (points.length <= 2) return points;
  const centroid = polygonCentroid(points);
  return points.map((point) => ({
    x: centroid.x + (point.x - centroid.x) * scale,
    y: centroid.y + (point.y - centroid.y) * scale,
  }));
}

function chaikinSmooth(points, iterations = 3) {
  if (points.length <= 3) return points;
  let current = [...points];
  for (let iter = 0; iter < iterations; iter += 1) {
    const next = [];
    for (let i = 0; i < current.length; i += 1) {
      const p0 = current[i];
      const p1 = current[(i + 1) % current.length];
      next.push({ x: 0.75 * p0.x + 0.25 * p1.x, y: 0.75 * p0.y + 0.25 * p1.y });
      next.push({ x: 0.25 * p0.x + 0.75 * p1.x, y: 0.25 * p0.y + 0.75 * p1.y });
    }
    current = next;
  }
  return current;
}

function buildHighlightPolygons(points) {
  if (!points || points.length < 3) return null;

  const hull = buildConvexHull(points);
  const outer = chaikinSmooth(expandPolygon(hull, 1.1), 3);
  const inner = chaikinSmooth(expandPolygon(hull, 1.035), 3);
  const core = chaikinSmooth(expandPolygon(hull, 0.99), 2);

  return { outer, inner, core };
}

function polygonToTrace(points, fillcolor, linecolor = "rgba(0, 212, 224, 0.14)") {
  return {
    type: "scatter",
    mode: "lines",
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    hoverinfo: "skip",
    showlegend: false,
    fill: "toself",
    fillcolor,
    line: {
      color: linecolor,
      width: 0.8,
      shape: "spline",
      smoothing: 0.85,
    },
  };
}

function UniverseAccordion({ title, children, open, onToggle, nested = false }) {
  return (
    <div className={`universe-accordion ${nested ? "nested" : ""}`}>
      <button
        type="button"
        className="universe-accordion-btn"
        onClick={onToggle}
        aria-expanded={open}
      >
        <span>{title}</span>
        <span className="universe-accordion-icon">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="universe-accordion-panel">{children}</div>}
    </div>
  );
}

function MethodologyModal({ open, onClose }) {
  const [clusteringOpen, setClusteringOpen] = useState(false);
  const [blockedPcaOpen, setBlockedPcaOpen] = useState(false);
  const [artificialVarianceOpen, setArtificialVarianceOpen] = useState(false);
  const [seasonStandardizationOpen, setSeasonStandardizationOpen] = useState(false);
  const [playersRemovedOpen, setPlayersRemovedOpen] = useState(false);
  const [kMeansOpen, setKMeansOpen] = useState(false);
  const [euclideanOpen, setEuclideanOpen] = useState(false);
  const [clusterCountOpen, setClusterCountOpen] = useState(false);
  const [inspirationsOpen, setInspirationsOpen] = useState(false);

  if (!open) return null;

  return (
    <div
      className="universe-modal-backdrop universe-page-backdrop"
      role="presentation"
      onMouseDown={(event) => event.stopPropagation()}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <section
        className="universe-modal universe-page"
        role="dialog"
        aria-modal="true"
        aria-labelledby="universe-modal-title"
        onClick={(event) => event.stopPropagation()}
        onMouseDown={(event) => event.stopPropagation()}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header className="universe-modal-header universe-page-header">
          <div>
            <h2 id="universe-modal-title">What Is The Galaxy?</h2>
          </div>
          <button
            type="button"
            className="universe-modal-close-btn"
            onClick={onClose}
            aria-label="Close Galaxy explanation"
          >
            ×
          </button>
        </header>

        <div className="universe-modal-body universe-page-body">
          <div className="universe-page-shell">
            <section className="universe-hero-card">
              <p>
                My NBA Galaxy is a <span className="universe-cyan-text">3-D UMAP Embedding</span> of <span className="universe-cyan-text">every NBA player-season since 2016-17</span>, built to visualize statistical play-style similarity between players rather than player production alone. Point guards through centers are all in the same model. Each point represents one player-season, and distance between points is meant to reflect similarity across 77 different shooting, mid-range, rim pressure, playmaking, defensive, and playtype metrics.
              </p>
              <p>
                Nearby points are statistically similar player-seasons, cluster colors represent different <span className="universe-cyan-text">archetypes</span>, and <span className="universe-cyan-text">constellation lines</span> show nearest-player relationships in my full blocked-feature space.
              </p>
              <p>
                The Galaxy does the best job it can in preserving my <span className="universe-cyan-text">77-D feature space</span> in just <span className="universe-cyan-text">three dimensions</span>. This UMAP 3D Embedding is just for visual interpretation, while the actual similarity rankings are performed on the updated raw equal-block weighted feature space.
              </p>
            </section>

            <UniverseAccordion
              title="Learn about the clustering algorithm"
              open={clusteringOpen}
              onToggle={() => setClusteringOpen((previousValue) => !previousValue)}
            >
              <p className="universe-methodology-lede">
                I separated every player-season into 16 stable and interpretable archetypes using a <span className="universe-cyan-text">blocked-PCA</span>, <span className="universe-cyan-text">K-Means++ algorithm</span>. The archetypes are built from how a player actually plays, not from his listed position, which is why a stretch four and a stretch five can share one archetype while two players both listed at centre can land in completely different ones.
              </p>

              <UniverseAccordion
                title="Players Removed"
                open={playersRemovedOpen}
                onToggle={() => setPlayersRemovedOpen((previousValue) => !previousValue)}
                nested
              >
                <p>
                  <span className="universe-cyan-text">Nobody.</span> Every player-season that clears the minutes and games-played cut is in the model.
                </p>
                <p>
                  Earlier versions of the Galaxy held out LeBron James, Ben Simmons, and Scottie Barnes. That was never really about those three players; it was about the model being guards-only, which left three forwards being measured against a peer group they did not belong to. Now that every position is in the same model, the reason is gone and so is the exclusion.
                </p>
              </UniverseAccordion>

              <div className="universe-method-steps">
                <section className="universe-step-card">
                  <div className="universe-step-number">01</div>
                  <div className="universe-step-content">
                    <p>
                      I first standardized each player-season with respect to <span className="universe-cyan-text">every player in that same season</span>, not against players who share his listed position. Then I clipped each standardized feature at ±3.50 z-scores.
                    </p>
                    <UniverseAccordion
                      title="Why standardized W.R.T. each season?"
                      open={seasonStandardizationOpen}
                      onToggle={() => setSeasonStandardizationOpen((previousValue) => !previousValue)}
                      nested
                    >
                      <p>
                        Because NBA environments change over time, every feature is evaluated relative to the players of that same season. This prevents older seasons from being unfairly compared to modern seasons with different spacing, pace, 3-point volume, and offensive style.
                      </p>
                      <p>
                        Standardizing against the whole league rather than against a position group is deliberate. A centre who takes six threes a game is genuinely unusual, and the model should see that as unusual. Comparing him only to other centres would hide exactly the thing that makes his play style distinctive.
                      </p>
                    </UniverseAccordion>
                  </div>
                </section>

                <section className="universe-step-card">
                  <div className="universe-step-number">02</div>
                  <div className="universe-step-content">
                    <p>
                      Instead of applying PCA to all 77 features at once, I apply PCA inside each block separately. Each block is then weighted before clustering, so no single area of the game can dominate the entire archetype structure regardless of how many columns it happens to contain.
                    </p>
                    <UniverseAccordion
                      title="Why blocked-PCA over vanilla PCA?"
                      open={blockedPcaOpen}
                      onToggle={() => setBlockedPcaOpen((previousValue) => !previousValue)}
                      nested
                    >
                      <p>
                        Vanilla PCA would be horrible for this clustering scenario as it enables the linear interactions between completely unrelated features, preserving <span className="universe-red-text">artificial variance</span>.
                      </p>
                      <p>
                        My Blocked-PCA on the other hand enables only the features within each block to linearly interact with each other. Theoretically and experimentally, this forms substantially more meaningful principal components. The six blocks used are <span className="universe-cyan-text">Three-Point Shooting</span> (18 features), <span className="universe-cyan-text">Midrange Scoring</span> (10), <span className="universe-cyan-text">Rim Pressure</span> (10), <span className="universe-cyan-text">Playmaking</span> (10), <span className="universe-cyan-text">Defense</span> (7), and <span className="universe-cyan-text">Playtypes</span> (22). The first five each carry 15% of the final vector and Playtypes carries 25%.
                      </p>

                      <UniverseAccordion
                        title="What is artificial variance?"
                        open={artificialVarianceOpen}
                        onToggle={() => setArtificialVarianceOpen((previousValue) => !previousValue)}
                        nested
                      >
                        <p>
                          Due to overfitting, vanilla PCA is theoretically able to create principal components from linear combinations between completely unrelated features. In practice, MOST of the principal components returned by vanilla PCA are based on real signal/patterns, but the more variance, you retain the more prone your model is to overfitting. This is not just my hypothesis; my clustering algorithm performed significantly better with blocked-PCA compared to using vanilla PCA.
                        </p>
                      </UniverseAccordion>
                    </UniverseAccordion>
                  </div>
                </section>

                <section className="universe-step-card">
                  <div className="universe-step-number">03</div>
                  <div className="universe-step-content">
                    <p>
                      Finally, I run Euclidean-Based K-Means++ on the transformed blocked-feature space to separate the player-seasons into 16 archetypes.
                    </p>
                    <UniverseAccordion
                      title="Why K-Means++?"
                      open={kMeansOpen}
                      onToggle={() => setKMeansOpen((previousValue) => !previousValue)}
                      nested
                    >
                      <p>
                        I also thought K-Means++ was far too simple to use as my clustering algorithm. But due to the prevalence of possession-level features in my feature space, and my blocked-PCA idea, K-Means++ performed much better than HDBSCAN, DBSCAN, and Gaussian Mixture Models.
                      </p>
                    </UniverseAccordion>
                    <UniverseAccordion
                      title="Why Euclidean distance?"
                      open={euclideanOpen}
                      onToggle={() => setEuclideanOpen((previousValue) => !previousValue)}
                      nested
                    >
                      <p>
                        Due to the heavy usage of possession-level frequency features and PCA-Blocking, I found Euclidean distance to work best only after the pipeline standardizes each season, clips extreme values, compresses each block with PCA, and applies equal block weighting. That prevents Euclidean distance from comparing raw statistics without context.
                      </p>
                    </UniverseAccordion>
                    <UniverseAccordion
                      title="Why 16 archetypes?"
                      open={clusterCountOpen}
                      onToggle={() => setClusterCountOpen((previousValue) => !previousValue)}
                      nested
                    >
                      <p>
                        I swept k from 10 to 28 and scored each one on cluster stability, measured by re-running the clustering on 80% subsamples and checking how often players landed together again.
                      </p>
                      <p>
                        Stability falls as k rises, so the raw numbers always favour small k. Below 16 though, the model stops telling the truth about big men: every non-shooting centre collapses into a single group, and the difference between a rim-running lob threat, a physical paint scorer, and a floor-spacing five disappears. Above 16, the extra clusters stop describing anything new and simply cut the wing population into near-identical slices.
                      </p>
                      <p>
                        16 is where the model still separates four genuinely different kinds of big and an interior playmaking hub while keeping the guard structure intact.
                      </p>
                    </UniverseAccordion>
                  </div>
                </section>
              </div>
            </UniverseAccordion>

            <UniverseAccordion
              title="Inspirations"
              open={inspirationsOpen}
              onToggle={() => setInspirationsOpen((previousValue) => !previousValue)}
            >
              <p>
                I took visual inspiration from draftballr's draft prospect galaxy. I wanted to mirror it for NBA players. Our clustering and embedding techniques used are completely different though.
              </p>
            </UniverseAccordion>
          </div>
        </div>
      </section>
    </div>
  );
}

function WelcomeModal() {
  const STORAGE_KEY = "nbagalaxy:welcomeDismissed";
  const [visible, setVisible] = useState(() => !sessionStorage.getItem(STORAGE_KEY));

  function dismiss() {
    sessionStorage.setItem(STORAGE_KEY, "1");
    setVisible(false);
  }

  if (!visible) return null;

  return (
    <div className="welcome-overlay" onClick={dismiss}>
      <div className="welcome-card" onClick={(e) => e.stopPropagation()}>
        <canvas className="welcome-stars-canvas" ref={(canvas) => {
          if (!canvas) return;
          const ctx = canvas.getContext("2d");
          const W = canvas.width = canvas.offsetWidth;
          const H = canvas.height = canvas.offsetHeight;
          const stars = Array.from({ length: 120 }, () => ({
            x: Math.random() * W,
            y: Math.random() * H,
            r: Math.random() * 1.5 + 0.4,
            phase: Math.random() * Math.PI * 2,
            speed: 0.4 + Math.random() * 0.8,
          }));
          let frame;
          function draw(t) {
            ctx.clearRect(0, 0, W, H);
            stars.forEach((s) => {
              const alpha = 0.30 + 0.35 * Math.sin(t * 0.001 * s.speed + s.phase);
              ctx.beginPath();
              ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
              ctx.fillStyle = `rgba(223, 243, 244, ${alpha})`;
              ctx.fill();
            });
            frame = requestAnimationFrame(draw);
          }
          frame = requestAnimationFrame(draw);
          canvas._cleanup = () => cancelAnimationFrame(frame);
        }} />

        <button className="welcome-close" onClick={dismiss} aria-label="Close">✕</button>

        <p className="welcome-headline">This website is best viewed from a computer.</p>

        <div className="welcome-icon">
          <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
        </div>

        <div className="welcome-updates">
          <p className="welcome-subline">Wings + Bigs are here!</p>
          <p className="welcome-status">
            The Galaxy now covers every NBA player-season since 2016-17 at every position, sorted into 16 new archetypes. Badges were rebuilt from scratch for the full league: every percentile is now taken against all players rather than guards only, and there are seven new badges covering rim protection, interior scoring, screening, and perimeter defense.
          </p>
        </div>
      </div>
    </div>
  );
}

function ReadMeModal({ open, onClose }) {
  const [openSections, setOpenSections] = useState({});

  if (!open) return null;

  const toggleSection = (sectionKey) => {
    setOpenSections((previousSections) => ({
      ...previousSections,
      [sectionKey]: !previousSections[sectionKey],
    }));
  };

  return (
    <div
      className="universe-modal-backdrop universe-page-backdrop"
      role="presentation"
      onMouseDown={(event) => event.stopPropagation()}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <section
        className="universe-modal universe-page readme-page"
        role="dialog"
        aria-modal="true"
        aria-labelledby="readme-modal-title"
        onClick={(event) => event.stopPropagation()}
        onMouseDown={(event) => event.stopPropagation()}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header className="universe-modal-header universe-page-header">
          <div>
            <h2 id="readme-modal-title">Read Me</h2>
          </div>
          <button
            type="button"
            className="universe-modal-close-btn"
            onClick={onClose}
            aria-label="Close read me"
          >
            ×
          </button>
        </header>

        <div className="universe-modal-body universe-page-body">
          <div className="universe-page-shell readme-page-shell">
            <UniverseAccordion
              title="Selecting a player in the galaxy"
              open={Boolean(openSections.selectingPlayer)}
              onToggle={() => toggleSection("selectingPlayer")}
            >
              <p>
                By selecting a player season in the galaxy, you are able to see his most <span className="universe-cyan-text">similar players</span> by <span className="universe-cyan-text">play-styles</span>. In the similarity page itself you are able to see the <span className="universe-cyan-text">3PT, Mid-Range, Playmaking</span>, and <span className="universe-cyan-text">Defensive</span> similarity scores individually between the selected player and his "doppelgangers".
              </p>
            </UniverseAccordion>

            <UniverseAccordion
              title="Viewing a player's career path"
              open={Boolean(openSections.careerPath)}
              onToggle={() => toggleSection("careerPath")}
            >
              <p>
                You are also able to see how a player's <span className="universe-cyan-text">role</span> changes across his career by clicking the 'CAREER PATH' button in the player profile. This tells you what <span className="universe-cyan-text">clusters/archetypes</span> he was assigned to each year of his career.
              </p>
            </UniverseAccordion>

            <UniverseAccordion
              title="Viewing a player's skill break-down percentiles"
              open={Boolean(openSections.skillBreakdown)}
              onToggle={() => toggleSection("skillBreakdown")}
            >
              <p>
                Every player is also assigned an accurate <span className="universe-cyan-text">3-PT, Mid-Range, Rim Pressure, Playmaking</span> and <span className="universe-cyan-text">Defensive</span> skill percentile obtained through percentile calculations explained in the site. Players are also assigned <span className="universe-cyan-text">badges</span> based on their within-season percentiles of the medians of different groups of features.
              </p>
            </UniverseAccordion>

            <UniverseAccordion
              title="How badges work"
              open={Boolean(openSections.badges)}
              onToggle={() => toggleSection("badges")}
            >
              <p>
                There are <span className="universe-cyan-text">26 badges</span> across seven skill families: Three-Point Shooting, Mid-Range, Interior Scoring, Rim Pressure, Scoring, Playmaking, and Defense. Every badge score is a percentile taken against <span className="universe-cyan-text">every NBA player in that same season</span>, at every position.
              </p>
              <p>
                Ranking a centre against guards would normally be unfair in both directions, so each badge has an <span className="universe-cyan-text">opportunity gate</span>: you can only earn a badge in something you actually do at volume. A centre is never measured on pull-up three-point shooting because he never clears that gate, and a point guard is never measured on rim protection. The gates are always behavioural, based on what a player does on the floor, never on his listed position.
              </p>
              <p>
                Where volume and efficiency both matter, accuracy is compared <span className="universe-cyan-text">locally</span> rather than globally: a player's finishing is ranked against other players who shoot from that spot at a similar rate. The clearest case is Rim Protector, where opponent FG% difference is compared only against players who contest a similar share of shots. Rim contests are converted at a much higher rate than perimeter contests, so an unadjusted number would punish exactly the players doing the most defensive work.
              </p>
              <UniverseAccordion
                title="What the tiers mean"
                open={Boolean(openSections.badgeTiers)}
                onToggle={() => toggleSection("badgeTiers")}
                nested
              >
                <p>
                  Thresholds are not hand-picked. They are solved so that every badge lands on roughly the same share of the league, which stops a badge from becoming ordinary just because the skill underneath it is common.
                </p>
                <p>
                  <span className="universe-cyan-text">Bronze</span> is about the top 12% of the league at that skill, <span className="universe-cyan-text">Silver</span> the top 6%, <span className="universe-cyan-text">Gold</span> the top 2.4%, and <span className="universe-cyan-text">Diamond</span> roughly the top 0.6%.
                </p>
                <p>
                  Badges are meant to mark players who are genuinely great to elite at something. Plenty of real rotation players have <span className="universe-cyan-text">no badges at all</span>, and that is the intended result rather than a gap.
                </p>
              </UniverseAccordion>
            </UniverseAccordion>

            <UniverseAccordion
              title="Viewing Archetypes in The Galaxy"
              open={Boolean(openSections.viewingArchetypes)}
              onToggle={() => toggleSection("viewingArchetypes")}
            >
              <p>
                By clicking on any of the colored dots on the top bar, you are able to view the <span className="universe-cyan-text">minimum spanning tree</span> of the selected archetype as well as the archetype's <span className="universe-cyan-text">medoid player</span>.
              </p>
            </UniverseAccordion>

            <UniverseAccordion
              title="Viewing Features/Statistics of Archetypes"
              open={Boolean(openSections.archetypeFeatures)}
              onToggle={() => toggleSection("archetypeFeatures")}
            >
              <p>
                Upon clicking on the 'View Cluster Description' button when in cluster-selection view, you will enter a page displaying the <span className="universe-cyan-text">heat map</span> of all the players in the given cluster as well as a <span className="universe-cyan-text">text description</span> describing the <span className="universe-cyan-text">role</span> of cluster.
              </p>
            </UniverseAccordion>
          </div>
        </div>
      </section>
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [featureFilter, setFeatureFilter] = useState("");
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("kmeans");
  const [selectedDistanceMetric, setSelectedDistanceMetric] = useState("euclidean");
  const [selectedVisualizationMode, setSelectedVisualizationMode] = useState("3d_galaxy");
  const [clusterCounts, setClusterCounts] = useState({ kmeans: 12 });
  const [algorithmMenuOpen, setAlgorithmMenuOpen] = useState(false);
  const [distanceMetricMenuOpen, setDistanceMetricMenuOpen] = useState(false);
  const [visualizationMenuOpen, setVisualizationMenuOpen] = useState(false);
  const [methodologyOpen, setMethodologyOpen] = useState(false);
  const [readMeOpen, setReadMeOpen] = useState(false);
  const [clusterData, setClusterData] = useState(null);
  const [highlightedCluster, setHighlightedCluster] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [playerSearch, setPlayerSearch] = useState("");
  const [playerSearchOpen, setPlayerSearchOpen] = useState(false);
  const [activeSearchIndex, setActiveSearchIndex] = useState(-1);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [showAllFeatures, setShowAllFeatures] = useState(false);
  const [galaxyPlayerProfileHidden, setGalaxyPlayerProfileHidden] = useState(false);
  const [galaxyHoverTooltip, setGalaxyHoverTooltip] = useState(null);
  const [galaxyBestWorstOpen, setGalaxyBestWorstOpen] = useState(false);
  const [showPlayerNames, setShowPlayerNames] = useState(false);
  const [galaxyArchetypesEnabled, setGalaxyArchetypesEnabled] = useState(false);
  const [galaxyFullscreenEnabled] = useState(true);
  const [browserFullscreenActive, setBrowserFullscreenActive] = useState(false);
  const [galaxyCameraRevision, setGalaxyCameraRevision] = useState(0);
  const [plotAxisRange, setPlotAxisRange] = useState(null);
  const [loadingClusters, setLoadingClusters] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [hoveredSelectorFeature, setHoveredSelectorFeature] = useState(null);
  const [hoveredSelectorRect, setHoveredSelectorRect] = useState(null);
  const [selectorTooltipVisible, setSelectorTooltipVisible] = useState(false);
  const [selectorTooltipSize, setSelectorTooltipSize] = useState({ width: 0, height: 0 });
  const [glossaryMounted, setGlossaryMounted] = useState(false);
  const [glossaryVisible, setGlossaryVisible] = useState(false);
  const [activeCenterView, setActiveCenterView] = useState("plot");
  const [careerPathPlayerName, setCareerPathPlayerName] = useState("");
  const [selectedCareerMissingSeason, setSelectedCareerMissingSeason] = useState(null);
  const [selectedClusterReport, setSelectedClusterReport] = useState(null);
  const [loadingClusterReport, setLoadingClusterReport] = useState(false);
  const [clusterReportError, setClusterReportError] = useState("");
  const [similarPlayersSourcePoint, setSimilarPlayersSourcePoint] = useState(null);
  const [similarPlayersData, setSimilarPlayersData] = useState(null);
  const [loadingSimilarPlayers, setLoadingSimilarPlayers] = useState(false);
  const [similarPlayersError, setSimilarPlayersError] = useState("");
  const [skillBreakdownData, setSkillBreakdownData] = useState(null);
  const [loadingSkillBreakdown, setLoadingSkillBreakdown] = useState(false);
  const [skillBreakdownError, setSkillBreakdownError] = useState("");
  const [threePtBreakdownData, setThreePtBreakdownData] = useState(null);
  const [loadingThreePtBreakdown, setLoadingThreePtBreakdown] = useState(false);
  const [threePtBreakdownError, setThreePtBreakdownError] = useState("");
  const [clusterSummaryStatMode, setClusterSummaryStatMode] = useState("median");
  const [clusterSummaryValueMode, setClusterSummaryValueMode] = useState("percentile");
  const [viewTransitionActive, setViewTransitionActive] = useState(false);
  const [startupLoaderVisible, setStartupLoaderVisible] = useState(true);
  const [glossarySectionsOpen, setGlossarySectionsOpen] = useState(() =>
    Object.fromEntries(GLOSSARY_SECTIONS.map((section) => [section.key, true]))
  );
  const [leftPanelWidth, setLeftPanelWidth] = useState(() =>
    getStoredNumber(`${PANEL_STORAGE_PREFIX}:left-panel-width`, DEFAULT_LEFT_PANEL_WIDTH)
  );
  const [rightPanelWidth, setRightPanelWidth] = useState(() => {
    const storedRightPanelWidth = getStoredNumber(`${PANEL_STORAGE_PREFIX}:right-panel-width`, DEFAULT_RIGHT_PANEL_WIDTH);
    return storedRightPanelWidth <= COLLAPSED_PANEL_THRESHOLD ? DEFAULT_RIGHT_PANEL_WIDTH : storedRightPanelWidth;
  });
  const [error, setError] = useState("");
  const requestCounter = useRef(0);
  const mainLayoutRef = useRef(null);
  const plotWrapRef = useRef(null);
  const pendingPlotWheelOpsRef = useRef([]);
  const plotWheelFrameRef = useRef(null);
  const resizeStateRef = useRef(null);
  const playerSearchRef = useRef(null);
  const playerSearchInputRef = useRef(null);
  const selectorTooltipRef = useRef(null);
  const algorithmMenuRef = useRef(null);
  const distanceMetricMenuRef = useRef(null);
  const visualizationMenuRef = useRef(null);
  const galaxyCameraRef = useRef(null);
  const galaxyCameraAnimationFrameRef = useRef(null);
  const galaxyCameraReturnRef = useRef(null);
  const galaxyInteractionActiveRef = useRef(false);
  const galaxyInteractionResumeTimerRef = useRef(null);
  const galaxyLastClickSelectionRef = useRef({ playerKey: null, timestamp: 0 });
  const galaxyLastCameraMoveAtRef = useRef(0);
  const selectorShowTimerRef = useRef(null);
  const selectorHideTimerRef = useRef(null);
  const selectorCleanupTimerRef = useRef(null);
  const glossaryOpenTimerRef = useRef(null);
  const glossaryCleanupTimerRef = useRef(null);
  const viewSwapTimerRef = useRef(null);
  const viewTransitionTimerRef = useRef(null);
  const playerDetailCacheRef = useRef(new Map());
  const clusterReportCacheRef = useRef(new Map());
  const clusterReportRequestIdRef = useRef(0);

  const clusterDescriptionViewEnabled = activeCenterView === "cluster_description";
  const careerPathViewEnabled = activeCenterView === "career_path";
  const similarPlayersViewEnabled = activeCenterView === "similar_players";
  const skillBreakdownViewEnabled = activeCenterView === "skill_breakdown";
  const threePtBreakdownViewEnabled = activeCenterView === "three_pt_breakdown";
  const playerReportIdleVisible = activeCenterView === "plot" && !selectedPoint && !selectedCareerMissingSeason;
  const nonScatterViewEnabled = clusterDescriptionViewEnabled || careerPathViewEnabled || similarPlayersViewEnabled || skillBreakdownViewEnabled || threePtBreakdownViewEnabled;
  const currentAlgorithm = "kmeans";
  const isFuzzyMode = false;
  const isEuclideanKmeansLockedMode =
    selectedAlgorithm === "kmeans" &&
    selectedDistanceMetric === "euclidean" &&
    Boolean(config?.euclidean_kmeans_locked_mode);
  const isFeatureLockedMode = isEuclideanKmeansLockedMode;
  const isClusterCountLockedMode = isEuclideanKmeansLockedMode;
  const lockedEuclideanKmeansK = config?.euclidean_kmeans_locked_k ?? 12;
  const unlockedActiveClusterCount = clusterCounts.kmeans ?? (config?.default_kmeans_k ?? config?.default_k ?? 12);
  const activeClusterCount = isEuclideanKmeansLockedMode
    ? lockedEuclideanKmeansK
    : unlockedActiveClusterCount;
  const euclideanKmeansLockedFeatures = config?.euclidean_kmeans_locked_features ?? [];
  const lockedFeatureList = isEuclideanKmeansLockedMode ? euclideanKmeansLockedFeatures : [];
  const requestFeatures = isFeatureLockedMode ? lockedFeatureList : selectedFeatures;
  const galaxyDataAvailable = Boolean(
    clusterData?.galaxy?.enabled
    && clusterData?.points?.some((point) => (
      Number.isFinite(Number(point.galaxy_x))
      && Number.isFinite(Number(point.galaxy_y))
      && Number.isFinite(Number(point.galaxy_z))
    ))
  );
  const galaxyPlotEnabled = selectedVisualizationMode === "3d_galaxy" && galaxyDataAvailable;
  const galaxyFullscreenPlotActive = galaxyFullscreenEnabled && galaxyPlotEnabled && activeCenterView === "plot";
  const drawerOpen = Boolean(showAllFeatures && (selectedDetail || selectedClusterReport) && !galaxyFullscreenPlotActive);
  const activeVisualizationLabel = activeCenterView === "plot"
    ? getVisualizationModeLabel(galaxyPlotEnabled ? "3d_galaxy" : "2d_pca")
    : getCenterViewLabel(activeCenterView);
  const defaultPlotAxisRange = useMemo(
    () => getDefaultPlotAxisRange(clusterData?.points ?? []),
    [clusterData]
  );
  const clusterReportRequestPayload = useMemo(() => {
    if (!clusterData || highlightedCluster == null) return null;
    return {
      algorithm: clusterData.algorithm,
      distance_metric: clusterData.distance_metric,
      k: clusterData.k,
      features: clusterData.selected_features,
      cluster_number: highlightedCluster,
    };
  }, [clusterData, highlightedCluster]);
  const clusterReportRequestKey = useMemo(() => {
    if (!clusterReportRequestPayload) return null;
    return buildClusterReportRequestKey(clusterReportRequestPayload);
  }, [clusterReportRequestPayload]);

  const stopPanelResize = () => {
    resizeStateRef.current = null;
    if (typeof document !== "undefined") {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
  };

  const startPanelResize = (side, event) => {
    const fullscreenRightResize = galaxyFullscreenEnabled && galaxyPlotEnabled && side === "right";
    if (typeof window !== "undefined" && window.innerWidth <= 1220 && !fullscreenRightResize) return;


    resizeStateRef.current = {
      side,
      startX: event.clientX,
      startWidth: side === "left" ? leftPanelWidth : rightPanelWidth,
    };

    if (typeof document !== "undefined") {
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    event.preventDefault();
  };

  const clearSelectorTooltipTimers = () => {
    window.clearTimeout(selectorShowTimerRef.current);
    window.clearTimeout(selectorHideTimerRef.current);
    window.clearTimeout(selectorCleanupTimerRef.current);
  };

  const clearGlossaryTimers = () => {
    window.clearTimeout(glossaryOpenTimerRef.current);
    window.clearTimeout(glossaryCleanupTimerRef.current);
  };

  const clearViewTransitionTimers = () => {
    window.clearTimeout(viewSwapTimerRef.current);
    window.clearTimeout(viewTransitionTimerRef.current);
  };

  const cancelGalaxyCameraAnimation = () => {
    if (galaxyCameraAnimationFrameRef.current == null) return;
    window.cancelAnimationFrame(galaxyCameraAnimationFrameRef.current);
    galaxyCameraAnimationFrameRef.current = null;
  };

  const cloneGalaxyCamera = (camera) => {
    const currentCamera = camera ?? GALAXY_DEFAULT_CAMERA;
    return {
      ...GALAXY_DEFAULT_CAMERA,
      ...currentCamera,
      eye: {
        x: Number(currentCamera.eye?.x ?? GALAXY_DEFAULT_CAMERA.eye.x),
        y: Number(currentCamera.eye?.y ?? GALAXY_DEFAULT_CAMERA.eye.y),
        z: Number(currentCamera.eye?.z ?? GALAXY_DEFAULT_CAMERA.eye.z),
      },
      center: {
        x: Number(currentCamera.center?.x ?? GALAXY_DEFAULT_CAMERA.center.x),
        y: Number(currentCamera.center?.y ?? GALAXY_DEFAULT_CAMERA.center.y),
        z: Number(currentCamera.center?.z ?? GALAXY_DEFAULT_CAMERA.center.z),
      },
      up: {
        x: Number(currentCamera.up?.x ?? GALAXY_DEFAULT_CAMERA.up.x),
        y: Number(currentCamera.up?.y ?? GALAXY_DEFAULT_CAMERA.up.y),
        z: Number(currentCamera.up?.z ?? GALAXY_DEFAULT_CAMERA.up.z),
      },
    };
  };

  const interpolateGalaxyVector = (startVector, endVector, progress) => ({
    x: Number(startVector.x ?? 0) + (Number(endVector.x ?? 0) - Number(startVector.x ?? 0)) * progress,
    y: Number(startVector.y ?? 0) + (Number(endVector.y ?? 0) - Number(startVector.y ?? 0)) * progress,
    z: Number(startVector.z ?? 0) + (Number(endVector.z ?? 0) - Number(startVector.z ?? 0)) * progress,
  });

  // Spherical linear interpolation for unit direction vectors — produces a smooth
  // great-circle arc so the camera doesn't cut through the galaxy on large rotations.
  const slerpGalaxyVector = (a, b, t) => {
    const cosAngle = Math.max(-1, Math.min(1,
      Number(a.x ?? 0) * Number(b.x ?? 0) +
      Number(a.y ?? 0) * Number(b.y ?? 0) +
      Number(a.z ?? 0) * Number(b.z ?? 0)
    ));
    if (Math.abs(cosAngle) >= 0.9999) return interpolateGalaxyVector(a, b, t);
    const angle = Math.acos(cosAngle);
    const sinAngle = Math.sin(angle);
    const fa = Math.sin((1 - t) * angle) / sinAngle;
    const fb = Math.sin(t * angle) / sinAngle;
    return { x: fa * a.x + fb * b.x, y: fa * a.y + fb * b.y, z: fa * a.z + fb * b.z };
  };

  // Quintic ease-in-out: smoother acceleration/deceleration for large camera rotations.
  const easeGalaxyCameraProgress = (progress) => {
    const t = clamp(progress, 0, 1);
    return t < 0.5
      ? 16 * t * t * t * t * t
      : 1 - ((-2 * t + 2) ** 5) / 2;
  };

  const animateGalaxyCameraTo = (targetCamera, durationMs = GALAXY_FOCUS_CAMERA_ANIMATION_MS) => {
    if (typeof window === "undefined") {
      galaxyCameraRef.current = cloneGalaxyCamera(targetCamera);
      setGalaxyCameraRevision((previousRevision) => previousRevision + 1);
      return;
    }

    cancelGalaxyCameraAnimation();
    const startCamera = cloneGalaxyCamera(galaxyCameraRef.current ?? GALAXY_DEFAULT_CAMERA);
    const endCamera = cloneGalaxyCamera(targetCamera);
    const startTime = window.performance.now();

    // Pre-decompose eye into direction + distance so we can slerp the arc.
    const startEyeDist = Math.hypot(startCamera.eye.x, startCamera.eye.y, startCamera.eye.z) || 1;
    const endEyeDist   = Math.hypot(endCamera.eye.x,   endCamera.eye.y,   endCamera.eye.z)   || 1;
    const startEyeDir  = { x: startCamera.eye.x / startEyeDist, y: startCamera.eye.y / startEyeDist, z: startCamera.eye.z / startEyeDist };
    const endEyeDir    = { x: endCamera.eye.x   / endEyeDist,   y: endCamera.eye.y   / endEyeDist,   z: endCamera.eye.z   / endEyeDist };

    const step = (currentTime) => {
      const rawProgress = durationMs <= 0 ? 1 : (currentTime - startTime) / durationMs;
      const easedProgress = easeGalaxyCameraProgress(rawProgress);
      const interpDir  = slerpGalaxyVector(startEyeDir, endEyeDir, easedProgress);
      const interpDist = startEyeDist + (endEyeDist - startEyeDist) * easedProgress;
      galaxyCameraRef.current = {
        ...GALAXY_DEFAULT_CAMERA,
        eye: { x: interpDir.x * interpDist, y: interpDir.y * interpDist, z: interpDir.z * interpDist },
        center: interpolateGalaxyVector(startCamera.center, endCamera.center, easedProgress),
        up: slerpGalaxyVector(
          normalizeGalaxyVector(startCamera.up),
          normalizeGalaxyVector(endCamera.up),
          easedProgress
        ),
      };
      setGalaxyCameraRevision((previousRevision) => previousRevision + 1);

      if (rawProgress < 1) {
        galaxyCameraAnimationFrameRef.current = window.requestAnimationFrame(step);
        return;
      }

      galaxyCameraAnimationFrameRef.current = null;
      galaxyCameraRef.current = endCamera;
      setGalaxyCameraRevision((previousRevision) => previousRevision + 1);
    };

    galaxyCameraAnimationFrameRef.current = window.requestAnimationFrame(step);
  };

  const beginGalaxyUserCameraInteraction = () => {
    if (typeof window === "undefined") return;
    cancelGalaxyCameraAnimation();
    galaxyInteractionActiveRef.current = true;
    window.clearTimeout(galaxyInteractionResumeTimerRef.current);
  };

  const releaseGalaxyUserCameraInteractionSoon = (delayMs = 180) => {
    if (typeof window === "undefined") {
      galaxyInteractionActiveRef.current = false;
      return;
    }

    window.clearTimeout(galaxyInteractionResumeTimerRef.current);
    galaxyInteractionResumeTimerRef.current = window.setTimeout(() => {
      galaxyInteractionActiveRef.current = false;
    }, delayMs);
  };

  const getGalaxyPointPosition = (point) => ({
    x: Number(point?.galaxy_x ?? point?.pc1 ?? 0),
    y: Number(point?.galaxy_y ?? point?.pc2 ?? 0),
    z: Number(point?.galaxy_z ?? 0),
  });

  const getGalaxyCoordinateFrame = (points) => {
    const positions = points.map(getGalaxyPointPosition);
    const xs = positions.map((position) => position.x);
    const ys = positions.map((position) => position.y);
    const zs = positions.map((position) => position.z);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    return {
      mid: {
        x: (minX + maxX) / 2,
        y: (minY + maxY) / 2,
        z: (minZ + maxZ) / 2,
      },
      span: {
        x: Math.max(maxX - minX, 1e-6),
        y: Math.max(maxY - minY, 1e-6),
        z: Math.max(maxZ - minZ, 1e-6),
      },
    };
  };

  const normalizeGalaxyPositionForCamera = (position, frame) => ({
    x: (position.x - frame.mid.x) / frame.span.x,
    y: (position.y - frame.mid.y) / frame.span.y,
    z: (position.z - frame.mid.z) / frame.span.z,
  });

  const normalizeGalaxyVector = (vector, fallback = { x: 1, y: 0, z: 0 }) => {
    const normalizedVector = {
      x: Number.isFinite(Number(vector?.x)) ? Number(vector.x) : Number(fallback.x ?? 1),
      y: Number.isFinite(Number(vector?.y)) ? Number(vector.y) : Number(fallback.y ?? 0),
      z: Number.isFinite(Number(vector?.z)) ? Number(vector.z) : Number(fallback.z ?? 0),
    };
    const length = Math.hypot(normalizedVector.x, normalizedVector.y, normalizedVector.z);
    if (length <= 1e-8) return normalizeGalaxyVector(fallback, { x: 1, y: 0, z: 0 });
    return {
      x: normalizedVector.x / length,
      y: normalizedVector.y / length,
      z: normalizedVector.z / length,
    };
  };

  const crossGalaxyVector = (a, b) => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  });

  const dotGalaxyVector = (a, b) => (
    Number(a.x ?? 0) * Number(b.x ?? 0)
    + Number(a.y ?? 0) * Number(b.y ?? 0)
    + Number(a.z ?? 0) * Number(b.z ?? 0)
  );

  const getStableGalaxyCameraAxes = (camera) => {
    const fallbackEye = GALAXY_DEFAULT_CAMERA.eye;
    const fallbackEyeDistance = Math.hypot(fallbackEye.x, fallbackEye.y, fallbackEye.z) || 1;
    const fallbackDirection = {
      x: fallbackEye.x / fallbackEyeDistance,
      y: fallbackEye.y / fallbackEyeDistance,
      z: fallbackEye.z / fallbackEyeDistance,
    };
    const rawEye = camera?.eye ?? fallbackEye;
    const rawEyeDistance = Math.hypot(Number(rawEye.x ?? 0), Number(rawEye.y ?? 0), Number(rawEye.z ?? 0));
    const viewDirection = normalizeGalaxyVector(
      rawEyeDistance > 1e-5 ? rawEye : fallbackEye,
      fallbackDirection
    );
    const safeViewDirection = Math.abs(viewDirection.z) >= 0.18
      ? viewDirection
      : normalizeGalaxyVector({
          x: viewDirection.x,
          y: viewDirection.y,
          z: viewDirection.z < 0 ? -0.18 : 0.18,
        }, fallbackDirection);
    const requestedUp = normalizeGalaxyVector(camera?.up ?? GALAXY_DEFAULT_CAMERA.up, GALAXY_DEFAULT_CAMERA.up);
    const fallbackUp = Math.abs(dotGalaxyVector(requestedUp, safeViewDirection)) > 0.92
      ? { x: 0, y: 1, z: 0 }
      : requestedUp;
    const right = normalizeGalaxyVector(crossGalaxyVector(fallbackUp, safeViewDirection), { x: 1, y: 0, z: 0 });
    const screenUp = normalizeGalaxyVector(crossGalaxyVector(safeViewDirection, right), GALAXY_DEFAULT_CAMERA.up);

    return {
      viewDirection: safeViewDirection,
      right,
      screenUp,
    };
  };

  const buildStableGalaxyEye = (currentEye, targetDistance) => {
    const { viewDirection } = getStableGalaxyCameraAxes({ eye: currentEye, up: GALAXY_DEFAULT_CAMERA.up });
    return {
      x: viewDirection.x * targetDistance,
      y: viewDirection.y * targetDistance,
      z: viewDirection.z * targetDistance,
    };
  };

  const getGalaxyFocusViewport = () => {
    if (typeof window === "undefined") {
      return { width: 1280, height: 720, aspectRatio: 16 / 9, sidebarBiasRatio: 0 };
    }

    const rect = plotWrapRef.current?.getBoundingClientRect?.();
    const width = Math.max(360, Number(rect?.width ?? window.innerWidth ?? 1280));
    const height = Math.max(320, Number(rect?.height ?? window.innerHeight ?? 720));
    const mainLayoutRect = mainLayoutRef.current?.getBoundingClientRect?.();
    const plotRightEdge = Number(rect?.right ?? width);
    const layoutRightEdge = Number(mainLayoutRect?.right ?? window.innerWidth ?? plotRightEdge);
    const hiddenRightWidth = Math.max(0, layoutRightEdge - plotRightEdge);
    const sidebarBiasRatio = hiddenRightWidth > 80 ? GALAXY_PLAYER_FOCUS_SIDEBAR_BIAS_RATIO : 0;

    return {
      width,
      height,
      aspectRatio: width / height,
      sidebarBiasRatio,
    };
  };

  // Zoom in on the selected player: keep the current camera direction exactly,
  // use a fixed close distance, but zoom out just enough if any comp would be off-screen.
  const computeSelectedPlayerGalaxyCamera = (pointToFocus, currentCamera) => {
    if (!pointToFocus || !clusterData?.points?.length) return null;

    const frame = getGalaxyCoordinateFrame(clusterData.points);
    const selectedPosition = normalizeGalaxyPositionForCamera(
      getGalaxyPointPosition(pointToFocus),
      frame,
    );

    // Preserve the current camera direction — no rotation, just zoom + pan.
    const safeCamera = currentCamera ?? GALAXY_PLAYER_FOCUS_BASE_CAMERA;
    const { viewDirection, screenUp } = getStableGalaxyCameraAxes(safeCamera);
    const viewport = getGalaxyFocusViewport();
    const right = normalizeGalaxyVector(crossGalaxyVector(screenUp, viewDirection));

    // Project the 4 comps onto the current view plane and find the minimum
    // camera distance needed to keep all of them within the viewport.
    const pointByKey = new Map(clusterData.points.map((p) => [String(p.player_key), p]));
    const similarEdges = Array.isArray(clusterData.galaxy?.similarity_edges) ? clusterData.galaxy.similarity_edges : [];
    const neighborPositions = similarEdges
      .filter((e) => String(e.source) === String(pointToFocus.player_key))
      .sort((a, b) => Number(a.rank) - Number(b.rank))
      .slice(0, GALAXY_SELECTED_NEIGHBOR_COUNT)
      .map((e) => pointByKey.get(String(e.target)))
      .filter(Boolean)
      .map((p) => normalizeGalaxyPositionForCamera(getGalaxyPointPosition(p), frame));

    const hFov = 2 * Math.atan(Math.tan(GALAXY_PLAYER_FOCUS_VERTICAL_FOV_RADIANS / 2) * viewport.aspectRatio);
    const pad = GALAXY_PLAYER_FOCUS_VIEWPORT_PADDING;
    let fittingDistance = GALAXY_PLAYER_FOCUS_ZOOM_DISTANCE;
    for (const np of neighborPositions) {
      const dx = dotGalaxyVector({ x: np.x - selectedPosition.x, y: np.y - selectedPosition.y, z: np.z - selectedPosition.z }, right);
      const dy = dotGalaxyVector({ x: np.x - selectedPosition.x, y: np.y - selectedPosition.y, z: np.z - selectedPosition.z }, screenUp);
      const dByX = Math.abs(dx) / (pad * Math.tan(hFov / 2));
      const dByY = Math.abs(dy) / (pad * Math.tan(GALAXY_PLAYER_FOCUS_VERTICAL_FOV_RADIANS / 2));
      fittingDistance = Math.max(fittingDistance, dByX, dByY);
    }
    const targetDistance = Math.min(fittingDistance, GALAXY_PLAYER_FOCUS_MAX_ZOOM_DISTANCE);

    // Shift center slightly left so the selected player clears the right sidebar.
    const center = {
      x: selectedPosition.x - right.x * viewport.sidebarBiasRatio * targetDistance,
      y: selectedPosition.y - right.y * viewport.sidebarBiasRatio * targetDistance,
      z: selectedPosition.z - right.z * viewport.sidebarBiasRatio * targetDistance,
    };

    return {
      ...GALAXY_DEFAULT_CAMERA,
      eye: {
        x: viewDirection.x * targetDistance,
        y: viewDirection.y * targetDistance,
        z: viewDirection.z * targetDistance,
      },
      center,
      up: screenUp,
    };
  };

  const focusGalaxyCameraOnPoint = (pointToFocus, options = {}) => {
    if (!galaxyPlotEnabled || !pointToFocus || !clusterData?.points?.length) return;

    const currentCamera = cloneGalaxyCamera(galaxyCameraRef.current ?? GALAXY_DEFAULT_CAMERA);
    if (options.storeReturnCamera !== false && !selectedPoint && !galaxyCameraReturnRef.current) {
      galaxyCameraReturnRef.current = currentCamera;
    }

    galaxyInteractionActiveRef.current = false;
    window.clearTimeout(galaxyInteractionResumeTimerRef.current);

    const targetCamera = computeSelectedPlayerGalaxyCamera(pointToFocus, currentCamera);
    if (!targetCamera) return;
    animateGalaxyCameraTo(targetCamera, options.durationMs ?? GALAXY_PLAYER_FOCUS_CAMERA_ANIMATION_MS);
  };

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    if (!selectedPoint || !galaxyPlotEnabled || activeCenterView !== "plot") return undefined;

    const refocusTimer = window.setTimeout(() => {
      focusGalaxyCameraOnPoint(selectedPoint, {
        storeReturnCamera: false,
        durationMs: 520,
      });
    }, 120);

    return () => window.clearTimeout(refocusTimer);
  }, [activeCenterView, galaxyPlotEnabled, galaxyFullscreenEnabled, rightPanelWidth, leftPanelWidth]);

  const focusGalaxyCameraOnCluster = (clusterNumber) => {
    if (!galaxyPlotEnabled || !clusterData?.points?.length) return;
    const numericCluster = Number(clusterNumber);
    if (!Number.isFinite(numericCluster)) return;

    const currentCamera = cloneGalaxyCamera(galaxyCameraRef.current ?? GALAXY_DEFAULT_CAMERA);
    if (!galaxyCameraReturnRef.current) {
      galaxyCameraReturnRef.current = currentCamera;
    }

    const clusterPoints = clusterData.points.filter((point) => Number(point.cluster) === numericCluster);
    if (!clusterPoints.length) return;

    const frame = getGalaxyCoordinateFrame(clusterData.points);
    const normalizedClusterPoints = clusterPoints.map((point) => normalizeGalaxyPositionForCamera(getGalaxyPointPosition(point), frame));
    const clusterCenter = normalizedClusterPoints.reduce(
      (accumulator, position) => ({
        x: accumulator.x + position.x / normalizedClusterPoints.length,
        y: accumulator.y + position.y / normalizedClusterPoints.length,
        z: accumulator.z + position.z / normalizedClusterPoints.length,
      }),
      { x: 0, y: 0, z: 0 }
    );

    const clusterRadius = Math.max(
      0.001,
      ...normalizedClusterPoints.map((position) => Math.sqrt(
        (position.x - clusterCenter.x) ** 2
        + (position.y - clusterCenter.y) ** 2
        + (position.z - clusterCenter.z) ** 2
      ))
    );
    const targetDistance = clamp(
      clusterRadius * GALAXY_CLUSTER_FOCUS_RADIUS_MULTIPLIER,
      GALAXY_CLUSTER_FOCUS_MIN_CAMERA_DISTANCE,
      GALAXY_CLUSTER_FOCUS_MAX_CAMERA_DISTANCE
    );

    animateGalaxyCameraTo({
      ...GALAXY_DEFAULT_CAMERA,
      eye: { x: targetDistance * 0.03, y: targetDistance * 0.05, z: targetDistance },
      center: clusterCenter,
      up: { x: 0, y: 1, z: 0 },
    });
  };

  const resetGalaxyCameraFocus = () => {
    if (!galaxyPlotEnabled) return;
    animateGalaxyCameraTo(galaxyCameraReturnRef.current ?? GALAXY_LAUNCH_CAMERA, 740);
    galaxyCameraReturnRef.current = null;
  };


  const applyCursorCenteredGalaxyWheelZoom = (event) => {
    if (!galaxyPlotEnabled || activeCenterView !== "plot") return;
    const rect = plotWrapRef.current?.getBoundingClientRect?.();
    if (!rect) return;
    const currentCamera = cloneGalaxyCamera(galaxyCameraRef.current ?? GALAXY_LAUNCH_CAMERA);
    const axes = getStableGalaxyCameraAxes(currentCamera);
    const currentEye = currentCamera.eye ?? GALAXY_LAUNCH_CAMERA.eye;
    const currentDistance = Math.max(1e-5, Math.hypot(Number(currentEye.x ?? 0), Number(currentEye.y ?? 0), Number(currentEye.z ?? 0)));
    const boundedDeltaY = clamp(Number(event.deltaY ?? 0), -180, 180);
    const zoomFactor = Math.exp(boundedDeltaY * 0.0019);
    const nextDistance = clamp(currentDistance * zoomFactor, GALAXY_MIN_CAMERA_DISTANCE, GALAXY_MAX_CAMERA_DISTANCE);
    const fractionX = clamp((Number(event.clientX ?? rect.left + rect.width / 2) - rect.left) / Math.max(1, rect.width), 0, 1);
    const fractionY = clamp((Number(event.clientY ?? rect.top + rect.height / 2) - rect.top) / Math.max(1, rect.height), 0, 1);
    const ndcX = (fractionX - 0.5) * 2;
    const ndcY = (0.5 - fractionY) * 2;
    const zoomInStrength = clamp(1 - zoomFactor, -0.16, 0.26);
    const panScale = currentDistance * Math.max(0.18, Math.min(0.52, 0.30 + currentDistance * 0.06));
    const centerShift = {
      x: (axes.right.x * ndcX + axes.screenUp.x * ndcY) * panScale * zoomInStrength,
      y: (axes.right.y * ndcX + axes.screenUp.y * ndcY) * panScale * zoomInStrength,
      z: (axes.right.z * ndcX + axes.screenUp.z * ndcY) * panScale * zoomInStrength,
    };
    const currentCenter = currentCamera.center ?? GALAXY_DEFAULT_CAMERA.center;
    galaxyCameraRef.current = {
      ...GALAXY_DEFAULT_CAMERA,
      ...currentCamera,
      center: {
        x: Number(currentCenter.x ?? 0) + centerShift.x,
        y: Number(currentCenter.y ?? 0) + centerShift.y,
        z: Number(currentCenter.z ?? 0) + centerShift.z,
      },
      eye: {
        x: axes.viewDirection.x * nextDistance,
        y: axes.viewDirection.y * nextDistance,
        z: axes.viewDirection.z * nextDistance,
      },
      up: axes.screenUp,
    };
    setGalaxyCameraRevision((previousRevision) => previousRevision + 1);
  };

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    if (!galaxyPlotEnabled) return undefined;

    const resizeHandle = window.setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
      setGalaxyCameraRevision((previousRevision) => previousRevision + 1);
    }, 80);

    return () => window.clearTimeout(resizeHandle);
  }, [galaxyFullscreenEnabled, galaxyPlotEnabled]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const layoutInitialized = window.localStorage.getItem(PANEL_LAYOUT_INIT_KEY);
    if (!layoutInitialized) {
      setLeftPanelWidth(DEFAULT_LEFT_PANEL_WIDTH);
      setRightPanelWidth(DEFAULT_RIGHT_PANEL_WIDTH);
      window.localStorage.setItem(PANEL_LAYOUT_INIT_KEY, "true");
    }
  }, []);

  useEffect(() => {
    return () => {
      clearSelectorTooltipTimers();
      clearGlossaryTimers();
      clearViewTransitionTimers();
      window.clearTimeout(galaxyInteractionResumeTimerRef.current);
      galaxyInteractionActiveRef.current = false;
      cancelGalaxyCameraAnimation();
      stopPanelResize();
    };
  }, []);

  useEffect(() => {
    if (!algorithmMenuOpen) return undefined;

    const handlePointerDown = (event) => {
      if (!algorithmMenuRef.current?.contains(event.target)) {
        setAlgorithmMenuOpen(false);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [algorithmMenuOpen]);

  useEffect(() => {
    if (!distanceMetricMenuOpen) return undefined;

    const handlePointerDown = (event) => {
      if (!distanceMetricMenuRef.current?.contains(event.target)) {
        setDistanceMetricMenuOpen(false);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [distanceMetricMenuOpen]);

  useEffect(() => {
    if (!visualizationMenuOpen) return undefined;

    const handlePointerDown = (event) => {
      if (!visualizationMenuRef.current?.contains(event.target)) {
        setVisualizationMenuOpen(false);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [visualizationMenuOpen]);

  useEffect(() => {
    if (!isFeatureLockedMode) return;
    clearSelectorTooltipTimers();
    setHoveredSelectorFeature(null);
    setSelectorTooltipVisible(false);
  }, [isFeatureLockedMode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(`${PANEL_STORAGE_PREFIX}:left-panel-width`, String(leftPanelWidth));
  }, [leftPanelWidth]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(`${PANEL_STORAGE_PREFIX}:right-panel-width`, String(rightPanelWidth));
  }, [rightPanelWidth]);

  useEffect(() => {
    setPlotAxisRange(null);
    cancelGalaxyCameraAnimation();
    galaxyCameraRef.current = null;
    galaxyCameraReturnRef.current = null;
  }, [clusterData]);


  useEffect(() => {
    if (typeof document === "undefined") return undefined;
    const handleFullscreenChange = () => {
      setBrowserFullscreenActive(Boolean(document.fullscreenElement));
    };
    handleFullscreenChange();
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const handleBrowserFullscreenToggle = async () => {
    if (typeof document === "undefined") return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch (fullscreenError) {
      console.warn("Browser fullscreen request failed", fullscreenError);
    }
  };

  useEffect(() => {
    if (!galaxyPlotEnabled || activeCenterView !== "plot" || selectedPoint || (galaxyArchetypesEnabled && highlightedCluster != null)) return;
    if (galaxyCameraRef.current) return;
    galaxyCameraRef.current = cloneGalaxyCamera(GALAXY_LAUNCH_CAMERA);
    setGalaxyCameraRevision((previousRevision) => previousRevision + 1);
  }, [activeCenterView, galaxyPlotEnabled, selectedPoint, galaxyArchetypesEnabled, highlightedCluster]);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    if (activeCenterView !== "plot") return undefined;
    if (galaxyPlotEnabled) return undefined;
    if (!defaultPlotAxisRange || !plotWrapRef.current) return undefined;

    const plotNode = plotWrapRef.current;

    const applyPendingWheelOperations = () => {
      plotWheelFrameRef.current = null;
      const operations = pendingPlotWheelOpsRef.current.splice(0);
      if (!operations.length) return;

      setPlotAxisRange((previousRange) => {
        let nextRange = previousRange ?? defaultPlotAxisRange;

        operations.forEach((operation) => {
          if (operation.type === "zoom") {
            nextRange = buildZoomedAxisRange(
              nextRange,
              defaultPlotAxisRange,
              operation.fractionX,
              operation.fractionYFromBottom,
              operation.zoomFactor
            );
          } else if (operation.type === "pan") {
            nextRange = buildPannedAxisRange(
              nextRange,
              defaultPlotAxisRange,
              operation.deltaX,
              operation.deltaY,
              operation.plotWidth,
              operation.plotHeight
            );
          }
        });

        return nextRange;
      });
    };

    const queuePlotWheelOperation = (operation) => {
      pendingPlotWheelOpsRef.current.push(operation);
      if (plotWheelFrameRef.current != null) return;
      plotWheelFrameRef.current = window.requestAnimationFrame(applyPendingWheelOperations);
    };

    const handlePlotWheel = (event) => {
      const rect = plotNode.getBoundingClientRect();
      const plotWidth = Math.max(1, rect.width - PLOT_LAYOUT_MARGIN.l - PLOT_LAYOUT_MARGIN.r);
      const plotHeight = Math.max(1, rect.height - PLOT_LAYOUT_MARGIN.t - PLOT_LAYOUT_MARGIN.b);
      const isPinchGesture = event.ctrlKey || event.metaKey;

      event.preventDefault();
      event.stopPropagation();

      if (isPinchGesture) {
        const fractionX = clamp(
          (event.clientX - rect.left - PLOT_LAYOUT_MARGIN.l) / plotWidth,
          0,
          1
        );
        const fractionYFromBottom = 1 - clamp(
          (event.clientY - rect.top - PLOT_LAYOUT_MARGIN.t) / plotHeight,
          0,
          1
        );
        const boundedDeltaY = clamp(event.deltaY, -PLOT_MAX_WHEEL_DELTA, PLOT_MAX_WHEEL_DELTA);
        const zoomFactor = Math.exp(boundedDeltaY * PLOT_PINCH_ZOOM_SENSITIVITY);

        queuePlotWheelOperation({
          type: "zoom",
          fractionX,
          fractionYFromBottom,
          zoomFactor,
        });
        return;
      }

      queuePlotWheelOperation({
        type: "pan",
        deltaX: event.deltaX,
        deltaY: event.deltaY,
        plotWidth,
        plotHeight,
      });
    };

    plotNode.addEventListener("wheel", handlePlotWheel, { passive: false });
    return () => {
      plotNode.removeEventListener("wheel", handlePlotWheel);
      pendingPlotWheelOpsRef.current = [];
      if (plotWheelFrameRef.current != null) {
        window.cancelAnimationFrame(plotWheelFrameRef.current);
        plotWheelFrameRef.current = null;
      }
    };
  }, [activeCenterView, defaultPlotAxisRange, galaxyPlotEnabled]);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    if (!galaxyPlotEnabled || activeCenterView !== "plot") return undefined;
    const plotNode = plotWrapRef.current;
    if (!plotNode) return undefined;

    const handlePointerDown = () => {
      beginGalaxyUserCameraInteraction();
    };

    const handlePointerUp = () => {
      releaseGalaxyUserCameraInteractionSoon(160);
    };

    const handleWheel = (event) => {
      event.preventDefault();
      event.stopPropagation();
      beginGalaxyUserCameraInteraction();
      applyCursorCenteredGalaxyWheelZoom(event);
      releaseGalaxyUserCameraInteractionSoon(260);
    };

    plotNode.addEventListener("pointerdown", handlePointerDown, { passive: true });
    plotNode.addEventListener("wheel", handleWheel, { passive: false });
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);

    return () => {
      plotNode.removeEventListener("pointerdown", handlePointerDown);
      plotNode.removeEventListener("wheel", handleWheel);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [activeCenterView, galaxyPlotEnabled]);



  useEffect(() => {
    const clampPanelSizes = () => {
      if (!mainLayoutRef.current) return;
      const fullscreenActive = galaxyFullscreenEnabled && galaxyPlotEnabled;
      if (typeof window !== "undefined" && window.innerWidth <= 1220 && !fullscreenActive) return;

      const { width } = mainLayoutRef.current.getBoundingClientRect();
      const { maxLeft, maxRight } = getPanelBounds(width, drawerOpen, fullscreenActive);

      if (!fullscreenActive) {
        setLeftPanelWidth((prev) => clamp(prev, MIN_LEFT_PANEL_WIDTH, maxLeft));
      }
      setRightPanelWidth((prev) => clamp(prev, MIN_RIGHT_PANEL_WIDTH, maxRight));

    };

    clampPanelSizes();
    window.addEventListener("resize", clampPanelSizes);
    return () => window.removeEventListener("resize", clampPanelSizes);
  }, [drawerOpen, galaxyFullscreenEnabled, galaxyPlotEnabled]);

  useEffect(() => {
    const handlePointerMove = (event) => {
      const resizeState = resizeStateRef.current;
      if (!resizeState || !mainLayoutRef.current) return;


      const { width } = mainLayoutRef.current.getBoundingClientRect();
      const fullscreenActive = galaxyFullscreenEnabled && galaxyPlotEnabled;
      const { maxLeft, maxRight } = getPanelBounds(width, drawerOpen, fullscreenActive);
      const deltaX = event.clientX - resizeState.startX;

      if (resizeState.side === "left") {
        if (!fullscreenActive) {
          setLeftPanelWidth(clamp(resizeState.startWidth + deltaX, MIN_LEFT_PANEL_WIDTH, maxLeft));
        }
        return;
      }

      setRightPanelWidth(clamp(resizeState.startWidth - deltaX, MIN_RIGHT_PANEL_WIDTH, maxRight));
      window.requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    };

    const handlePointerUp = () => {
      stopPanelResize();
    };

    window.addEventListener("mousemove", handlePointerMove);
    window.addEventListener("mouseup", handlePointerUp);
    return () => {
      window.removeEventListener("mousemove", handlePointerMove);
      window.removeEventListener("mouseup", handlePointerUp);
    };
  }, [drawerOpen, galaxyFullscreenEnabled, galaxyPlotEnabled]);

  useEffect(() => {
    let cancelled = false;

    const applyConfig = (data) => {
      setConfig(data);
      setSelectedFeatures(data.default_features ?? data.allowed_features ?? []);
      setSelectedAlgorithm("kmeans");
      setSelectedDistanceMetric("euclidean");
      setSelectedVisualizationMode("3d_galaxy");
      setClusterCounts({
        kmeans: data.euclidean_kmeans_locked_k ?? data.default_kmeans_k ?? data.default_k ?? 12,
      });
    };

    const loadStartupData = async () => {
      try {
        let bootstrapPayload = null;
        let lastBootstrapError = null;
        for (const bootstrapUrl of DEFAULT_BOOTSTRAP_URLS) {
          try {
            const bootstrapResponse = await fetch(bootstrapUrl);
            if (!bootstrapResponse.ok) {
              throw new Error(`Bootstrap unavailable (${bootstrapResponse.status})`);
            }
            bootstrapPayload = await bootstrapResponse.json();
            break;
          } catch (bootstrapError) {
            lastBootstrapError = bootstrapError;
          }
        }
        if (!bootstrapPayload) {
          throw lastBootstrapError ?? new Error("Bootstrap unavailable");
        }
        const bootstrapConfig = bootstrapPayload?.config;
        const bootstrapCluster = bootstrapPayload?.cluster;
        if (!bootstrapConfig || !bootstrapCluster?.points?.length) {
          throw new Error("Bootstrap payload is missing config or cluster data.");
        }
        if (cancelled) return;

        const clusterRequestPayload = {
          algorithm: bootstrapCluster.algorithm,
          distance_metric: bootstrapCluster.distance_metric,
          k: bootstrapCluster.k,
          features: bootstrapCluster.selected_features,
        };
        playerDetailCacheRef.current = new Map(
          Object.entries(bootstrapPayload.player_details_by_key ?? {}).map(([playerKey, detail]) => [
            String(playerKey),
            detail,
          ])
        );

        Object.entries(bootstrapPayload.cluster_reports_by_number ?? {}).forEach(([clusterNumber, report]) => {
          const reportKey = buildClusterReportRequestKey({
            ...clusterRequestPayload,
            cluster_number: Number(clusterNumber),
          });
          clusterReportCacheRef.current.set(reportKey, report);
        });

        setError("");
        setLoadingClusters(false);
        setClusterData(bootstrapCluster);
        applyConfig(bootstrapConfig);
      } catch {
        try {
          const response = await fetch(`${API_BASE}/api/config`);
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "Config request failed.");
          }
          if (!cancelled) {
            applyConfig(data);
          }
        } catch (err) {
          if (!cancelled) {
            setError(`Failed to load config: ${String(err)}`);
          }
        }
      }
    };

    loadStartupData();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!config || requestFeatures.length === 0) return;

    const requestPayload = {
      algorithm: selectedAlgorithm,
      distance_metric: selectedDistanceMetric,
      k: activeClusterCount,
      features: requestFeatures,
    };
    const requestedClusterKey = buildClusterRequestKey(requestPayload);
    const activeClusterDataKey = clusterData ? buildClusterRequestKey({
      algorithm: clusterData.algorithm,
      distance_metric: clusterData.distance_metric,
      k: clusterData.k,
      features: clusterData.selected_features,
    }) : "";
    if (clusterData && activeClusterDataKey === requestedClusterKey) {
      setLoadingClusters(false);
      return;
    }

    const currentRequest = ++requestCounter.current;
    const handle = setTimeout(async () => {
      setLoadingClusters(true);
      setError("");

      try {
        const res = await fetch(`${API_BASE}/api/cluster`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestPayload),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Cluster request failed.");
        if (currentRequest !== requestCounter.current) return;

        setClusterData(data);
        setHighlightedCluster((prev) => {
          if (prev == null) return prev;
          const clusterExists = data.cluster_sizes?.some((item) => item.cluster === prev);
          return clusterExists ? prev : null;
        });
        if (selectedPoint) {
          const refreshedPoint = data.points.find(
            (p) => p.player_key === selectedPoint.player_key
          );
          setSelectedPoint(refreshedPoint || null);
        }
      } catch (err) {
        if (currentRequest !== requestCounter.current) return;
        setError(String(err));
      } finally {
        if (currentRequest === requestCounter.current) {
          setLoadingClusters(false);
        }
      }
    }, 350);

    return () => clearTimeout(handle);
  }, [config, selectedAlgorithm, selectedDistanceMetric, activeClusterCount, requestFeatures, clusterData]);

  useEffect(() => {
    if (!selectedPoint) {
      setSelectedDetail(null);
      if (!clusterDescriptionViewEnabled) {
        setShowAllFeatures(false);
      }
      return;
    }

    const cachedDetail = playerDetailCacheRef.current.get(String(selectedPoint.player_key));
    if (cachedDetail) {
      setSelectedDetail(cachedDetail);
      setLoadingDetail(false);
      return;
    }

    let cancelled = false;
    setLoadingDetail(true);

    fetchStaticAsset(`/precomputed/players/${playerAssetSlug(selectedPoint.player_key)}.json`)
      .then((asset) => {
        if (asset?.detail) return asset.detail;
        return fetch(`${API_BASE}/api/player-details`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ player_key: selectedPoint.player_key }),
        }).then(async (response) => {
        // Without the !response.ok guard an error body -- {"detail": "Player row
        // not found."} from a backend whose dataset predates this player -- was
        // cached and rendered as if it were a detail panel, and the missing
        // fields threw during render, blanking the whole app.
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || 'Player detail request failed.');
          }
          return data;
        });
      })
      .then((data) => {
        if (!cancelled) {
          playerDetailCacheRef.current.set(String(selectedPoint.player_key), data);
          setSelectedDetail(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setSelectedDetail(null);
          setError(`Failed to load player detail: ${String(err)}`);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedPoint, clusterDescriptionViewEnabled]);

  useEffect(() => {
    setGalaxyBestWorstOpen(false);
    setShowAllFeatures(false);
    setGalaxyHoverTooltip(null);
  }, [selectedPoint?.player_key]);

  useEffect(() => {
    if (!clusterReportRequestPayload || !clusterReportRequestKey) {
      setSelectedClusterReport(null);
      setLoadingClusterReport(false);
      setClusterReportError("");
      if (!clusterDescriptionViewEnabled) {
        setShowAllFeatures(false);
      }
      return;
    }

    const cachedReport = clusterReportCacheRef.current.get(clusterReportRequestKey);
    if (cachedReport) {
      setSelectedClusterReport(cachedReport);
      setLoadingClusterReport(false);
      setClusterReportError("");
      return;
    }

    let cancelled = false;
    const requestId = clusterReportRequestIdRef.current + 1;
    clusterReportRequestIdRef.current = requestId;
    setSelectedClusterReport(null);
    setLoadingClusterReport(clusterDescriptionViewEnabled);
    setClusterReportError("");

    fetchStaticAsset(`/precomputed/cluster_reports/${clusterReportRequestPayload.cluster_number}.json`)
      .then((asset) => {
        if (asset) return asset;
        return fetch(`${API_BASE}/api/cluster-report`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(clusterReportRequestPayload),
        }).then(async (response) => {
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || 'Cluster report request failed.');
          }
          return data;
        });
      })
      .then((data) => {
        if (!cancelled && clusterReportRequestIdRef.current === requestId) {
          clusterReportCacheRef.current.set(clusterReportRequestKey, data);
          setSelectedClusterReport(data);
        }
      })
      .catch((err) => {
        if (!cancelled && clusterReportRequestIdRef.current === requestId) {
          setSelectedClusterReport(null);
          setClusterReportError(String(err));
        }
      })
      .finally(() => {
        if (!cancelled && clusterReportRequestIdRef.current === requestId) {
          setLoadingClusterReport(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [clusterDescriptionViewEnabled, clusterReportRequestPayload, clusterReportRequestKey]);

  useEffect(() => {
    if (!startupLoaderVisible) return;
    if (config && !loadingClusters && (clusterData || error)) {
      setStartupLoaderVisible(false);
    }
  }, [startupLoaderVisible, config, loadingClusters, clusterData, error]);

  useEffect(() => {
    if (!similarPlayersViewEnabled || !similarPlayersSourcePoint) {
      if (!similarPlayersViewEnabled) {
        setSimilarPlayersError("");
      }
      return;
    }

    let cancelled = false;
    setLoadingSimilarPlayers(true);
    setSimilarPlayersError("");

    fetchStaticAsset(`/precomputed/comps/${playerAssetSlug(similarPlayersSourcePoint.player_key)}.json`)
      .then((asset) => {
        if (asset) return asset;
        return fetch(buildSimilarPlayersUrl({
          sourcePoint: similarPlayersSourcePoint,
          clusterData,
          config,
          activeClusterCount,
        })).then(async (response) => {
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "Similar players request failed.");
          }
          return data;
        });
      })
      .then((data) => {
        if (!cancelled) {
          setSimilarPlayersData(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setSimilarPlayersData(null);
          setSimilarPlayersError(String(err));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingSimilarPlayers(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [similarPlayersViewEnabled, similarPlayersSourcePoint, clusterData, config, activeClusterCount]);


  useEffect(() => {
    if (!skillBreakdownViewEnabled || !selectedPoint) {
      if (!skillBreakdownViewEnabled) {
        setSkillBreakdownError("");
      }
      return undefined;
    }

    let cancelled = false;
    setLoadingSkillBreakdown(true);
    setSkillBreakdownError("");

    fetchStaticAsset(`/precomputed/players/${playerAssetSlug(selectedPoint.player_key)}.json`)
      .then((asset) => {
        if (asset?.skill) return asset.skill;
        return fetch(`${API_BASE}/api/player-skill-breakdown`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            algorithm: clusterData?.algorithm ?? selectedAlgorithm,
            distance_metric: clusterData?.distance_metric ?? selectedDistanceMetric,
            k: clusterData?.k ?? activeClusterCount,
            features: clusterData?.selected_features ?? requestFeatures,
            player_key: selectedPoint.player_key,
            cluster_number: selectedPoint.cluster,
          }),
        }).then(async (response) => {
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "Skill breakdown request failed.");
          }
          return data;
        });
      })
      .then((data) => {
        if (!cancelled) {
          setSkillBreakdownData(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setSkillBreakdownData(null);
          setSkillBreakdownError(String(err));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingSkillBreakdown(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [skillBreakdownViewEnabled, selectedPoint, clusterData, selectedAlgorithm, selectedDistanceMetric, activeClusterCount, requestFeatures]);

  useEffect(() => {
    if (!threePtBreakdownViewEnabled || !selectedPoint) {
      if (!threePtBreakdownViewEnabled) {
        setThreePtBreakdownError("");
      }
      return undefined;
    }

    let cancelled = false;
    setLoadingThreePtBreakdown(true);
    setThreePtBreakdownError("");

    fetchStaticAsset(`/precomputed/players/${playerAssetSlug(selectedPoint.player_key)}.json`)
      .then((asset) => {
        if (asset?.three_pt) return asset.three_pt;
        return fetch(`${API_BASE}/api/player-three-pt-breakdown`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            algorithm: clusterData?.algorithm ?? selectedAlgorithm,
            distance_metric: clusterData?.distance_metric ?? selectedDistanceMetric,
            k: clusterData?.k ?? activeClusterCount,
            features: clusterData?.selected_features ?? requestFeatures,
            player_key: selectedPoint.player_key,
            cluster_number: selectedPoint.cluster,
          }),
        }).then(async (response) => {
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "3PT breakdown request failed.");
          }
          return data;
        });
      })
      .then((data) => {
        if (!cancelled) {
          setThreePtBreakdownData(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setThreePtBreakdownData(null);
          setThreePtBreakdownError(String(err));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingThreePtBreakdown(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [threePtBreakdownViewEnabled, selectedPoint, clusterData, selectedAlgorithm, selectedDistanceMetric, activeClusterCount, requestFeatures]);

  useEffect(() => {
    if (!hoveredSelectorFeature) return undefined;

    const updateTooltipSize = () => {
      if (!selectorTooltipRef.current) return;
      const rect = selectorTooltipRef.current.getBoundingClientRect();
      const nextWidth = Math.ceil(rect.width);
      const nextHeight = Math.ceil(rect.height);

      setSelectorTooltipSize((prev) => {
        if (prev.width === nextWidth && prev.height === nextHeight) {
          return prev;
        }
        return { width: nextWidth, height: nextHeight };
      });
    };

    updateTooltipSize();
    window.addEventListener("resize", updateTooltipSize);
    return () => window.removeEventListener("resize", updateTooltipSize);
  }, [hoveredSelectorFeature, hoveredSelectorRect]);

  useEffect(() => {
    if (!glossaryMounted) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setGlossaryVisible(false);
        clearGlossaryTimers();
        glossaryCleanupTimerRef.current = window.setTimeout(() => {
          setGlossaryMounted(false);
        }, GLOSSARY_EXIT_MS);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [glossaryMounted]);

  useEffect(() => {
    if (!methodologyOpen && !readMeOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setMethodologyOpen(false);
        setReadMeOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [methodologyOpen, readMeOpen]);

  useEffect(() => {
    if (galaxyPlotEnabled && showPlayerNames) {
      setShowPlayerNames(false);
    }
  }, [galaxyPlotEnabled, showPlayerNames]);

  useEffect(() => {
    const handlePointerDown = (event) => {
      if (!playerSearchRef.current?.contains(event.target)) {
        setPlayerSearchOpen(false);
        setActiveSearchIndex(-1);
      }
    };

    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, []);

  const searchablePlayers = useMemo(() => {
    if (!clusterData?.points) return [];

    return clusterData.points.map((point) => {
      const haystack = normalizeSearchText(
        `${point.player_name} ${point.season} ${point.teams_played} ${point.position}`
      );
      const playerOnly = normalizeSearchText(point.player_name);
      return {
        point,
        haystack,
        playerOnly,
      };
    });
  }, [clusterData]);

  const playerSearchResults = useMemo(() => {
    const query = normalizeSearchText(playerSearch);
    if (!query) return [];

    const tokens = query.split(" ").filter(Boolean);
    return searchablePlayers
      .filter((entry) => tokens.every((token) => entry.haystack.includes(token)))
      .map((entry) => {
        const exactMatch = entry.haystack === query ? 0 : 1;
        const startsWithName = entry.playerOnly.startsWith(query) ? 0 : 1;
        const startsWithAny = entry.haystack.startsWith(query) ? 0 : 1;
        const indexScore = entry.haystack.indexOf(query);
        return {
          ...entry,
          score: (
            exactMatch * 1000
            + startsWithName * 200
            + startsWithAny * 80
            + (indexScore >= 0 ? indexScore : 999)
            + entry.point.player_name.length * 0.01
          ),
        };
      })
      .sort((a, b) => a.score - b.score || a.point.player_name.localeCompare(b.point.player_name) || a.point.season.localeCompare(b.point.season))
      .slice(0, 8);
  }, [playerSearch, searchablePlayers]);

  useEffect(() => {
    setActiveSearchIndex((prev) => {
      if (!playerSearchResults.length) return -1;
      if (prev < 0) return 0;
      return Math.min(prev, playerSearchResults.length - 1);
    });
  }, [playerSearchResults]);

  const selectGalaxyPoint = (point, options = {}) => {
    if (!point) return;
    const { blurSearch = false, updateCareerPath = false } = options;
    galaxyInteractionActiveRef.current = false;
    window.clearTimeout(galaxyInteractionResumeTimerRef.current);
    setGalaxyHoverTooltip(null);
    setGalaxyArchetypesEnabled(false);
    setHighlightedCluster(null);
    setSelectedPoint(point);
    setSelectedCareerMissingSeason(null);
    if (updateCareerPath || activeCenterView === "career_path") {
      setCareerPathPlayerName(point.player_name);
    }
    setPlayerSearch(formatPlayerSearchValue(point));
    setPlayerSearchOpen(false);
    setActiveSearchIndex(-1);
    if (blurSearch) playerSearchInputRef.current?.blur();
    if (galaxyPlotEnabled && activeCenterView === "plot") {
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => focusGalaxyCameraOnPoint(point));
      });
    }
  };

  const handlePlayerSearchSelect = (point) => {
    selectGalaxyPoint(point, { blurSearch: true });
  };

  const handlePlayerSearchChange = (event) => {
    setPlayerSearch(event.target.value);
    setPlayerSearchOpen(true);
    setActiveSearchIndex(0);
  };

  const handlePlayerSearchKeyDown = (event) => {
    if (event.key === "Escape") {
      setPlayerSearchOpen(false);
      setActiveSearchIndex(-1);
      return;
    }

    if (!playerSearchResults.length) return;

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setPlayerSearchOpen(true);
      setActiveSearchIndex((prev) => (prev + 1) % playerSearchResults.length);
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setPlayerSearchOpen(true);
      setActiveSearchIndex((prev) => (prev <= 0 ? playerSearchResults.length - 1 : prev - 1));
      return;
    }

    if (event.key === "Enter") {
      const nextIndex = activeSearchIndex >= 0 ? activeSearchIndex : 0;
      const nextEntry = playerSearchResults[nextIndex];
      if (nextEntry) {
        event.preventDefault();
        handlePlayerSearchSelect(nextEntry.point);
      }
    }
  };

  const transitionToCenterView = (nextView, options = {}) => {
    const immediate = Boolean(options?.immediate);
    if (viewTransitionActive || activeCenterView === nextView) return;

    clearViewTransitionTimers();
    setViewTransitionActive(true);

    const swapDelayMs = immediate ? 0 : VIEW_GLITCH_SWAP_MS;
    viewSwapTimerRef.current = window.setTimeout(() => {
      setActiveCenterView(nextView);
    }, swapDelayMs);

    viewTransitionTimerRef.current = window.setTimeout(() => {
      setViewTransitionActive(false);
    }, VIEW_GLITCH_TOTAL_MS);
  };

  const handleOpenClusterDescription = () => {
    if (highlightedCluster == null) return;
    if (!selectedClusterReport) {
      setLoadingClusterReport(true);
    }
    transitionToCenterView("cluster_description", { immediate: true });
  };

  const handleOpenCareerPath = () => {
    if (!selectedPoint) return;
    setCareerPathPlayerName(selectedPoint.player_name);
    setSelectedCareerMissingSeason(null);
    setGalaxyBestWorstOpen(false);
    setShowAllFeatures(false);
    transitionToCenterView("career_path");
  };

  const handleOpenSimilarPlayers = () => {
    if (!selectedPoint) return;
    setSimilarPlayersSourcePoint(selectedPoint);
    setSimilarPlayersData(null);
    setSimilarPlayersError("");
    setSelectedCareerMissingSeason(null);
    setGalaxyBestWorstOpen(false);
    setShowAllFeatures(false);
    transitionToCenterView("similar_players");
  };


  const handleOpenSkillBreakdown = () => {
    if (!selectedPoint) return;
    setSkillBreakdownData(null);
    setSkillBreakdownError("");
    setSelectedCareerMissingSeason(null);
    setGalaxyBestWorstOpen(false);
    setShowAllFeatures(false);
    transitionToCenterView("skill_breakdown");
  };

  const handleOpenThreePtBreakdown = () => {
    if (!selectedPoint) return;
    setThreePtBreakdownData(null);
    setThreePtBreakdownError("");
    setSelectedCareerMissingSeason(null);
    setShowAllFeatures(false);
    transitionToCenterView("three_pt_breakdown");
  };

  const handleSelectSimilarPlayer = (similarPlayer) => {
    if (!similarPlayer || !clusterData?.points?.length) return;
    const targetName = normalizeComparableValue(similarPlayer.player_name);
    const targetSeason = normalizeComparableValue(similarPlayer.season);
    const targetTeam = normalizeComparableValue(similarPlayer.team);
    const matchedPoint = clusterData.points.find((point) => (
      normalizeComparableValue(point.player_name) === targetName
      && normalizeComparableValue(point.season) === targetSeason
      && (!targetTeam || normalizeComparableValue(point.teams_played).includes(targetTeam) || targetTeam.includes(normalizeComparableValue(point.teams_played)))
    )) ?? clusterData.points.find((point) => (
      normalizeComparableValue(point.player_name) === targetName
      && normalizeComparableValue(point.season) === targetSeason
    ));

    if (!matchedPoint) {
      setSimilarPlayersError(`Could not find ${similarPlayer.player_name} ${similarPlayer.season} in the active galaxy view.`);
      return;
    }

    setShowAllFeatures(false);
    selectGalaxyPoint(matchedPoint);
  };

  const handleBackToGalaxy = () => {
    setSelectedCareerMissingSeason(null);
    transitionToCenterView("plot");
  };

  const visibleFeatures = useMemo(() => {
    if (!config) return [];

    const featurePool = isFeatureLockedMode ? lockedFeatureList : config.allowed_features;
    const q = featureFilter.trim().toLowerCase();
    if (!q) return featurePool;

    return featurePool.filter((feature) => {
      const meta = getFeatureMeta(feature);
      return `${feature} ${meta.label}`.toLowerCase().includes(q);
    });
  }, [config, featureFilter, isFeatureLockedMode, lockedFeatureList]);

  const displayedSelectedFeatures = isFeatureLockedMode ? lockedFeatureList : selectedFeatures;

  const clusterLegendItems = useMemo(() => {
    if (!clusterData?.cluster_sizes) return [];
    return clusterData.cluster_sizes.map((item) => ({
      cluster: item.cluster,
      count: item.count,
      color: getClusterColor(item.cluster),
      name: getConfigClusterName(config, item.cluster, clusterData.algorithm, clusterData.distance_metric),
    }));
  }, [clusterData, config]);

  const highlightedClusterName = useMemo(() => {
    if (highlightedCluster == null) return "";
    return (
      clusterLegendItems.find((item) => item.cluster === highlightedCluster)?.name
      || getConfigClusterName(config, highlightedCluster, currentAlgorithm, selectedDistanceMetric)
      || `Cluster ${highlightedCluster}`
    );
  }, [clusterLegendItems, highlightedCluster, config, currentAlgorithm, selectedDistanceMetric]);

  const highlightedClusterCount = useMemo(() => {
    if (highlightedCluster == null) return 0;
    return clusterLegendItems.find((item) => Number(item.cluster) === Number(highlightedCluster))?.count ?? 0;
  }, [clusterLegendItems, highlightedCluster]);

  const careerPathTimeline = useMemo(() => {
    if (!clusterData?.points?.length || !careerPathPlayerName) return [];

    const targetPlayerName = normalizeSearchText(careerPathPlayerName);
    const playerPoints = clusterData.points
      .filter((point) => normalizeSearchText(point.player_name) === targetPlayerName)
      .sort((a, b) => getSeasonStartYear(a.season) - getSeasonStartYear(b.season));

    if (!playerPoints.length) return [];

    const allScatterSeasons = sortSeasons([...new Set(clusterData.points.map((point) => point.season))]);
    const qualifiedSeasonLookup = new Map();
    playerPoints.forEach((point) => {
      if (!qualifiedSeasonLookup.has(point.season)) {
        qualifiedSeasonLookup.set(point.season, point);
      }
    });

    const qualifiedIndices = playerPoints
      .map((point) => allScatterSeasons.indexOf(point.season))
      .filter((index) => index >= 0);

    if (!qualifiedIndices.length) return [];

    const firstIndex = Math.min(...qualifiedIndices);
    const lastIndex = Math.max(...qualifiedIndices);

    return allScatterSeasons.slice(firstIndex, lastIndex + 1).map((season) => {
      const point = qualifiedSeasonLookup.get(season);
      if (point) {
        return {
          season,
          qualified: true,
          point,
          cluster: point.cluster,
          cluster_name: getConfigClusterName(config, point.cluster, clusterData.algorithm, clusterData.distance_metric),
        };
      }

      return {
        season,
        qualified: false,
        player_name: careerPathPlayerName,
      };
    });
  }, [clusterData, careerPathPlayerName, config]);

  const traces = useMemo(() => {
    if (!clusterData) return [];

    const byCluster = new Map();
    for (const point of clusterData.points) {
      if (!byCluster.has(point.cluster)) byCluster.set(point.cluster, []);
      byCluster.get(point.cluster).push(point);
    }

    const sortedEntries = Array.from(byCluster.entries()).sort((a, b) => a[0] - b[0]);
    const getPointColor = (cluster) => getClusterColor(cluster);

    if (galaxyPlotEnabled) {
      const pointByKey = new Map(clusterData.points.map((point) => [String(point.player_key), point]));
      const galaxy = clusterData.galaxy ?? {};
      const similarEdges = Array.isArray(galaxy.similarity_edges) ? galaxy.similarity_edges : [];
      const clusterEdges = Array.isArray(galaxy.cluster_edges) ? galaxy.cluster_edges : [];
      const archetypeLabels = Array.isArray(galaxy.archetype_labels) ? galaxy.archetype_labels : [];
      const gx = (point) => Number(point?.galaxy_x ?? point?.pc1 ?? 0);
      const gy = (point) => Number(point?.galaxy_y ?? point?.pc2 ?? 0);
      const gz = (point) => Number(point?.galaxy_z ?? 0);
      const clusterArchetypeFocusActive = galaxyArchetypesEnabled && highlightedCluster != null;
      const selectedClusterPoints = clusterArchetypeFocusActive
        ? clusterData.points.filter((point) => Number(point.cluster) === Number(highlightedCluster))
        : [];
      const selectedClusterKeySet = new Set(selectedClusterPoints.map((point) => String(point.player_key)));
      const selectedClusterRawCenter = selectedClusterPoints.length
        ? selectedClusterPoints.reduce(
            (accumulator, point) => ({
              x: accumulator.x + gx(point) / selectedClusterPoints.length,
              y: accumulator.y + gy(point) / selectedClusterPoints.length,
              z: accumulator.z + gz(point) / selectedClusterPoints.length,
            }),
            { x: 0, y: 0, z: 0 }
          )
        : { x: 0, y: 0, z: 0 };
      const selectedClusterRawRadius = selectedClusterPoints.length
        ? Math.max(
            0.001,
            ...selectedClusterPoints.map((point) => Math.hypot(
              gx(point) - selectedClusterRawCenter.x,
              gy(point) - selectedClusterRawCenter.y,
              gz(point) - selectedClusterRawCenter.z
            ))
          )
        : 0.001;
      const selectedSimilarityEdges = selectedPoint
        ? similarEdges
            .filter((edge) => String(edge.source) === String(selectedPoint.player_key))
            .sort((a, b) => Number(a.rank) - Number(b.rank))
            .slice(0, GALAXY_SELECTED_NEIGHBOR_COUNT)
        : [];
      const selectedTargetKeySet = new Set(selectedSimilarityEdges.map((edge) => String(edge.target)));
      const selectedTargetPoints = selectedSimilarityEdges
        .map((edge) => pointByKey.get(String(edge.target)))
        .filter(Boolean);
      const galaxyFocusActive = Boolean(selectedPoint || clusterArchetypeFocusActive);
      const selectedPulse = 1;
      const neighborPulse = 1;

      const frozenFocusKeySet = new Set([
        ...(selectedPoint?.player_key ? [String(selectedPoint.player_key)] : []),
        ...selectedTargetKeySet,
        ...selectedClusterKeySet,
      ]);
      const getStablePhase = (seedValue) => {
        const seed = String(seedValue ?? "");
        let hash = 0;
        for (let i = 0; i < seed.length; i += 1) {
          hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
        }
        return Math.abs(hash % 6283) / 1000;
      };
      const applyFocusedAmbientDrift = (position, seedValue, shouldFreeze = false) => {
        if (!galaxyFocusActive || shouldFreeze) return position;
        const phase = getStablePhase(seedValue);
        const driftRadius = clusterArchetypeFocusActive ? 0.0032 : 0.0026;
        const angle = phase;
        return {
          x: position.x + Math.cos(angle) * driftRadius,
          y: position.y + Math.sin(angle * 0.83 + phase) * driftRadius * 0.72,
          z: position.z,
        };
      };
      const getAnimatedGalaxyCoords = (point) => applyFocusedAmbientDrift(
        { x: gx(point), y: gy(point), z: gz(point) },
        point?.player_key,
        frozenFocusKeySet.has(String(point?.player_key))
      );
      const getAnimatedGalaxyLabelCoords = (label) => applyFocusedAmbientDrift({
        x: Number(label?.x ?? 0),
        y: Number(label?.y ?? 0),
        z: Number(label?.z ?? 0),
      }, `label-${label?.cluster ?? ""}`, true);
      const tx = (point) => getAnimatedGalaxyCoords(point).x;
      const ty = (point) => getAnimatedGalaxyCoords(point).y;
      const tz = (point) => getAnimatedGalaxyCoords(point).z;
      const getGalaxyHoverText = (point) => {
        const clusterName = getConfigClusterName(
          config,
          point.cluster,
          clusterData.algorithm,
          clusterData.distance_metric
        );
        const dominantLine = `Cluster ${point.cluster} · ${clusterName}`;
        return `${formatPointSeasonName(point)}<br>${point.season} · ${point.teams_played} · ${point.position}<br>${dominantLine}`;
      };

      const activeConstellationClusterSet = new Set(
        galaxyArchetypesEnabled
          ? (highlightedCluster != null
              ? [Number(highlightedCluster)]
              : sortedEntries.map(([cluster]) => Number(cluster)))
          : []
      );
      const activeArchetypeLabels = archetypeLabels.filter((label) => (
        galaxyArchetypesEnabled
          ? (highlightedCluster != null ? Number(label.cluster) === Number(highlightedCluster) : true)
          : false
      ));
      const clusterMedoids = Array.isArray(galaxy.cluster_medoids) ? galaxy.cluster_medoids : [];
      const selectedClusterMedoid = clusterArchetypeFocusActive
        ? clusterMedoids.find((medoid) => Number(medoid.cluster) === Number(highlightedCluster))
        : null;
      const selectedClusterMedoidPoint = selectedClusterMedoid?.player_key
        ? pointByKey.get(String(selectedClusterMedoid.player_key))
        : null;

      const selectedClusterMedoidDistancePoint = selectedClusterMedoidPoint ?? selectedClusterPoints[0] ?? null;
      const allArchetypeOverviewActive = galaxyArchetypesEnabled && highlightedCluster == null;
      const archetypeOverviewKeySets = new Map();
      if (allArchetypeOverviewActive) {
        const medoidsByCluster = new Map(clusterMedoids.map((medoid) => [Number(medoid.cluster), medoid]));
        const labelsByCluster = new Map(archetypeLabels.map((label) => [Number(label.cluster), label]));
        sortedEntries.forEach(([cluster, points]) => {
          const numericCluster = Number(cluster);
          const medoid = medoidsByCluster.get(numericCluster);
          const medoidPoint = medoid?.player_key ? pointByKey.get(String(medoid.player_key)) : null;
          const label = labelsByCluster.get(numericCluster);
          const fallbackCenter = points.length
            ? points.reduce(
                (accumulator, point) => ({
                  x: accumulator.x + gx(point) / points.length,
                  y: accumulator.y + gy(point) / points.length,
                  z: accumulator.z + gz(point) / points.length,
                }),
                { x: 0, y: 0, z: 0 }
              )
            : { x: 0, y: 0, z: 0 };
          const center = medoidPoint
            ? { x: gx(medoidPoint), y: gy(medoidPoint), z: gz(medoidPoint) }
            : label
              ? { x: Number(label.x ?? fallbackCenter.x), y: Number(label.y ?? fallbackCenter.y), z: Number(label.z ?? fallbackCenter.z) }
              : fallbackCenter;
          const selectedKeys = points
            .map((point) => ({
              point,
              distance: Math.hypot(gx(point) - center.x, gy(point) - center.y, gz(point) - center.z),
            }))
            .sort((a, b) => a.distance - b.distance)
            .slice(0, GALAXY_ARCHETYPE_OVERVIEW_MST_NODE_LIMIT)
            .map(({ point }) => String(point.player_key));
          if (medoidPoint?.player_key) {
            selectedKeys.push(String(medoidPoint.player_key));
          }
          archetypeOverviewKeySets.set(numericCluster, new Set(selectedKeys));
        });
      }
      const selectedClusterConstellationKeySet = selectedClusterPoints.length > GALAXY_MAX_VISIBLE_CLUSTER_CONSTELLATION_POINTS && selectedClusterMedoidDistancePoint
        ? new Set(
            selectedClusterPoints
              .map((point) => ({
                point,
                distance: Math.hypot(
                  gx(point) - gx(selectedClusterMedoidDistancePoint),
                  gy(point) - gy(selectedClusterMedoidDistancePoint),
                  gz(point) - gz(selectedClusterMedoidDistancePoint)
                ),
              }))
              .sort((a, b) => a.distance - b.distance)
              .slice(0, GALAXY_MAX_VISIBLE_CLUSTER_CONSTELLATION_POINTS)
              .map(({ point }) => String(point.player_key))
          )
        : null;
      if (selectedClusterMedoidDistancePoint && selectedClusterConstellationKeySet) {
        selectedClusterConstellationKeySet.add(String(selectedClusterMedoidDistancePoint.player_key));
      }

      const clusterConstellationTraces = sortedEntries
        .filter(([cluster]) => activeConstellationClusterSet.has(Number(cluster)))
        .flatMap(([cluster]) => {
          const clusterColor = getClusterColor(cluster);
          const overviewConstellationKeySet = archetypeOverviewKeySets.get(Number(cluster));
          const x = [];
          const y = [];
          const z = [];
          clusterEdges
            .filter((edge) => Number(edge.cluster) === Number(cluster))
            .forEach((edge) => {
              if (
                clusterArchetypeFocusActive
                && selectedClusterConstellationKeySet
                && (!selectedClusterConstellationKeySet.has(String(edge.source)) || !selectedClusterConstellationKeySet.has(String(edge.target)))
              ) {
                return;
              }
              if (
                allArchetypeOverviewActive
                && overviewConstellationKeySet
                && (!overviewConstellationKeySet.has(String(edge.source)) || !overviewConstellationKeySet.has(String(edge.target)))
              ) {
                return;
              }
              const sourcePoint = pointByKey.get(String(edge.source));
              const targetPoint = pointByKey.get(String(edge.target));
              if (!sourcePoint || !targetPoint) return;
              const sourcePosition = getAnimatedGalaxyCoords(sourcePoint);
              const targetPosition = getAnimatedGalaxyCoords(targetPoint);
              x.push(sourcePosition.x, targetPosition.x, null);
              y.push(sourcePosition.y, targetPosition.y, null);
              z.push(sourcePosition.z, targetPosition.z, null);
            });
          if (!x.length) return [];
          const layerBase = {
            type: "scatter3d",
            mode: "lines",
            showlegend: false,
            x,
            y,
            z,
            hoverinfo: "skip",
          };
          const archetypeOverviewLineLayers = allArchetypeOverviewActive
            ? [
                { name: `Cluster ${cluster} constellation outer glow`, color: hexToRgba(clusterColor, 0.035), width: 3.2 },
                { name: `Cluster ${cluster} constellation halo`, color: hexToRgba(clusterColor, 0.09), width: 1.65 },
                { name: `Cluster ${cluster} constellation core`, color: hexToRgba(clusterColor, 0.42), width: 0.82 },
                { name: `Cluster ${cluster} constellation starlight`, color: "rgba(245, 253, 255, 0.12)", width: 0.34 },
              ]
            : [
                { name: `Cluster ${cluster} constellation outer glow`, color: hexToRgba(clusterColor, 0.105), width: 8.5 },
                { name: `Cluster ${cluster} constellation halo`, color: hexToRgba(clusterColor, 0.22), width: 4.6 },
                { name: `Cluster ${cluster} constellation core`, color: hexToRgba(clusterColor, 0.78), width: 1.85 },
                { name: `Cluster ${cluster} constellation starlight`, color: "rgba(245, 253, 255, 0.26)", width: 0.72 },
              ];
          return archetypeOverviewLineLayers.map((layer) => ({
            ...layerBase,
            name: layer.name,
            line: { color: layer.color, width: layer.width },
          }));
        });

      const similarityTrace = selectedPoint && selectedSimilarityEdges.length > 0
        ? selectedSimilarityEdges
            .flatMap((edge) => {
              const targetPoint = pointByKey.get(String(edge.target));
              if (!targetPoint) return [];
              const targetColor = getPointColor(targetPoint.cluster, targetPoint);
              const sourcePosition = getAnimatedGalaxyCoords(selectedPoint);
              const targetPosition = getAnimatedGalaxyCoords(targetPoint);
              const lineCoordinates = {
                x: [sourcePosition.x, targetPosition.x, null],
                y: [sourcePosition.y, targetPosition.y, null],
                z: [sourcePosition.z, targetPosition.z, null],
              };
              return [
                {
                  type: "scatter3d",
                  mode: "lines",
                  name: `Similarity nebula outer path ${edge.rank ?? ""}`,
                  showlegend: false,
                  ...lineCoordinates,
                  hoverinfo: "skip",
                  line: {
                    color: hexToRgba(targetColor, 0.075),
                    width: 18.5,
                  },
                },
                {
                  type: "scatter3d",
                  mode: "lines",
                  name: `Similarity nebula mid path ${edge.rank ?? ""}`,
                  showlegend: false,
                  ...lineCoordinates,
                  hoverinfo: "skip",
                  line: {
                    color: hexToRgba(targetColor, 0.20),
                    width: 9.25,
                  },
                },
                {
                  type: "scatter3d",
                  mode: "lines",
                  name: `Similarity path ${edge.rank ?? ""}`,
                  showlegend: false,
                  ...lineCoordinates,
                  hoverinfo: "skip",
                  line: {
                    color: hexToRgba(targetColor, 0.84),
                    width: 3.15,
                  },
                },
                {
                  type: "scatter3d",
                  mode: "lines",
                  name: `Similarity starlight core ${edge.rank ?? ""}`,
                  showlegend: false,
                  ...lineCoordinates,
                  hoverinfo: "skip",
                  line: {
                    color: "rgba(245, 253, 255, 0.38)",
                    width: 1.05,
                  },
                },
              ];
            })
        : [];

      const glowTraces = sortedEntries.map(([cluster, pts]) => {
        const clusterColor = getClusterColor(cluster);
        const isHighlighted = highlightedCluster === cluster;
        return {
          type: "scatter3d",
          mode: "markers",
          name: `Cluster ${cluster} glow`,
          showlegend: false,
          x: pts.map(tx),
          y: pts.map(ty),
          z: pts.map(tz),
          hoverinfo: "skip",
          marker: {
            size: pts.map((point) => {
              if (selectedPoint?.player_key === point.player_key) return 12.0;
              if (selectedTargetKeySet.has(String(point.player_key))) return 10.2;
              return isHighlighted ? 11.4 : 8.8;
            }),
            color: pts.map((point) => getPointColor(cluster, point)),
            opacity: galaxyFocusActive
              ? (highlightedCluster == null ? 0.13 : isHighlighted ? 0.2 : 0.055)
              : (highlightedCluster == null ? 0.24 : isHighlighted ? 0.33 : 0.095),
            line: { color: hexToRgba(clusterColor, isHighlighted ? 0.15 : 0.08), width: 0.4 },
          },
        };
      });

      const baseTraces = sortedEntries.map(([cluster, pts]) => {
        const clusterColor = getClusterColor(cluster);
        const isHighlighted = highlightedCluster === cluster;
        return {
          type: "scatter3d",
          mode: showPlayerNames ? "markers+text" : "markers",
          name: `Cluster ${cluster}`,
          showlegend: false,
          x: pts.map(tx),
          y: pts.map(ty),
          z: pts.map(tz),
          text: pts.map(formatPointSeasonName),
          hovertext: pts.map(getGalaxyHoverText),
          customdata: pts.map((p) => p.player_key),
          hoverinfo: "none",
          textposition: "top center",
          textfont: {
            family: "JetBrains Mono, monospace",
            size: galaxyFocusActive ? 11 : 8.5,
            color: "rgba(223, 243, 244, 0.82)",
          },
          marker: {
            size: pts.map((point) => {
              if (selectedPoint?.player_key === point.player_key) return 8.1;
              if (selectedTargetKeySet.has(String(point.player_key))) return 7.4;
              return galaxyFocusActive ? 4.15 : 5.45;
            }),
            color: pts.map((point) => getPointColor(cluster, point)),
            opacity: galaxyFocusActive
              ? (highlightedCluster == null ? 0.42 : isHighlighted ? 0.54 : 0.22)
              : (highlightedCluster == null ? 0.84 : isHighlighted ? 0.96 : 0.36),
            line: {
              color: isHighlighted ? hexToRgba(clusterColor, 0.72) : "rgba(223,243,244,0.18)",
              width: isHighlighted ? 1.35 : 0.62,
            },
          },
        };
      });

      const generalHitboxPoints = galaxyFocusActive
        ? clusterData.points.filter((point) => !frozenFocusKeySet.has(String(point.player_key)))
        : clusterData.points;
      const hoverHitboxTraces = [{
        type: "scatter3d",
        mode: "markers",
        name: "Galaxy player hitbox",
        showlegend: false,
        x: generalHitboxPoints.map(tx),
        y: generalHitboxPoints.map(ty),
        z: generalHitboxPoints.map(tz),
        customdata: generalHitboxPoints.map((point) => point.player_key),
        hovertext: generalHitboxPoints.map(getGalaxyHoverText),
        hoverinfo: "none",
        marker: {
          size: galaxyFocusActive ? 16 : 15,
          color: "rgba(255,255,255,0.01)",
          opacity: 0.01,
          line: { color: "rgba(255,255,255,0)", width: 0 },
        },
      }];

      const selectedNeighborGlowTrace = selectedTargetPoints.length > 0
        ? [
            {
              type: "scatter3d",
              mode: "markers",
              name: "Closest neighbor soft nebula aura",
              showlegend: false,
              x: selectedTargetPoints.map(tx),
              y: selectedTargetPoints.map(ty),
              z: selectedTargetPoints.map(tz),
              hoverinfo: "skip",
              marker: {
                size: 58.0 * neighborPulse,
                color: selectedTargetPoints.map((point) => getPointColor(point.cluster, point)),
                opacity: 0.045,
                line: { color: "rgba(255,255,255,0)", width: 0 },
              },
            },
            {
              type: "scatter3d",
              mode: "markers",
              name: "Closest neighbor outer aura",
              showlegend: false,
              x: selectedTargetPoints.map(tx),
              y: selectedTargetPoints.map(ty),
              z: selectedTargetPoints.map(tz),
              hoverinfo: "skip",
              marker: {
                size: 38.0 * neighborPulse,
                color: selectedTargetPoints.map((point) => getPointColor(point.cluster, point)),
                opacity: 0.105,
                line: { color: "rgba(255,255,255,0)", width: 0 },
              },
            },
            {
              type: "scatter3d",
              mode: "markers",
              name: "Closest neighbor inner aura",
              showlegend: false,
              x: selectedTargetPoints.map(tx),
              y: selectedTargetPoints.map(ty),
              z: selectedTargetPoints.map(tz),
              hoverinfo: "skip",
              marker: {
                size: 22.5 * neighborPulse,
                color: selectedTargetPoints.map((point) => getPointColor(point.cluster, point)),
                opacity: 0.25,
                line: { color: "rgba(255,255,255,0)", width: 0 },
              },
            },
          ]
        : [];

      const selectedGlowTrace = selectedPoint
        ? [
            {
              type: "scatter3d",
              mode: "markers",
              name: "Selected player outer aura",
              showlegend: false,
              x: [tx(selectedPoint)],
              y: [ty(selectedPoint)],
              z: [tz(selectedPoint)],
              hoverinfo: "skip",
              marker: {
                size: 68.0 * selectedPulse,
                color: "#FFFFFF",
                opacity: 0.064,
                line: { color: "rgba(255,255,255,0)", width: 0 },
              },
            },
            {
              type: "scatter3d",
              mode: "markers",
              name: "Selected player inner aura",
              showlegend: false,
              x: [tx(selectedPoint)],
              y: [ty(selectedPoint)],
              z: [tz(selectedPoint)],
              hoverinfo: "skip",
              marker: {
                size: 36.0 * selectedPulse,
                color: "#7DF9FF",
                opacity: 0.24,
                line: { color: "rgba(255,255,255,0)", width: 0 },
              },
            },
          ]
        : [];

      const selectedNeighborTrace = selectedTargetPoints.length > 0
        ? [{
            type: "scatter3d",
            mode: "markers+text",
            name: "Closest connected comps",
            showlegend: false,
            x: selectedTargetPoints.map(tx),
            y: selectedTargetPoints.map(ty),
            z: selectedTargetPoints.map(tz),
            text: selectedTargetPoints.map((point, index) => `#${index + 1} ${formatPointSeasonName(point)}`),
            customdata: selectedTargetPoints.map((point) => point.player_key),
            hovertext: selectedTargetPoints.map((point) => `${getGalaxyHoverText(point)}<br>Connected comp`),
            hoverinfo: "none",
            textposition: "top center",
            textfont: {
              family: "JetBrains Mono, monospace",
              size: 17,
              color: "rgba(245, 253, 255, 0.99)",
            },
            marker: {
              size: 14.2 * neighborPulse,
              color: selectedTargetPoints.map((point) => getPointColor(point.cluster, point)),
              opacity: 0.96,
              line: { color: "rgba(245,253,255,0.92)", width: 2.0 },
            },
          }]
        : [];

      const selectedNeighborHitboxTrace = selectedTargetPoints.length > 0
        ? [{
            type: "scatter3d",
            mode: "markers",
            name: "Selected neighbor hitbox",
            showlegend: false,
            x: selectedTargetPoints.map(tx),
            y: selectedTargetPoints.map(ty),
            z: selectedTargetPoints.map(tz),
            customdata: selectedTargetPoints.map((point) => point.player_key),
            hovertext: selectedTargetPoints.map((point) => `${getGalaxyHoverText(point)}<br>Connected comp`),
            hoverinfo: "none",
            marker: {
              size: 36,
              color: "rgba(255,255,255,0.01)",
              opacity: 0.01,
              line: { color: "rgba(255,255,255,0)", width: 0 },
            },
          }]
        : [];

      const selectedTrace = selectedPoint
        ? [{
            type: "scatter3d",
            mode: "markers+text",
            name: "Selected player",
            showlegend: false,
            x: [tx(selectedPoint)],
            y: [ty(selectedPoint)],
            z: [tz(selectedPoint)],
            text: [formatPointSeasonName(selectedPoint)],
            hovertext: [`${getGalaxyHoverText(selectedPoint)}<br>Selected player`],
            customdata: [selectedPoint.player_key],
            hoverinfo: "none",
            textposition: "top center",
            textfont: { family: "JetBrains Mono, monospace", size: 20, color: "rgba(255,255,255,0.99)" },
            marker: {
              size: 17.6 * selectedPulse,
              color: "#FFFFFF",
              opacity: 1,
              line: { color: "rgba(0,212,224,0.96)", width: 2.65 },
            },
          }]
        : [];

      const selectedClusterMedoidSunTrace = selectedClusterMedoidPoint
        ? (() => {
            const selectedClusterColor = getClusterColor(highlightedCluster);
            const medoidLabel = `${formatPointSeasonName(selectedClusterMedoidPoint)}<br>CLUSTER MEDOID`;
            return [
              {
                type: "scatter3d",
                mode: "markers",
                name: "Selected cluster medoid outer aura",
                showlegend: false,
                x: [tx(selectedClusterMedoidPoint)],
                y: [ty(selectedClusterMedoidPoint)],
                z: [tz(selectedClusterMedoidPoint)],
                hoverinfo: "skip",
                marker: {
                  size: 68.0 * selectedPulse,
                  color: "#FFFFFF",
                  opacity: 0.066,
                  line: { color: "rgba(255,255,255,0)", width: 0 },
                },
              },
              {
                type: "scatter3d",
                mode: "markers",
                name: "Selected cluster medoid inner aura",
                showlegend: false,
                x: [tx(selectedClusterMedoidPoint)],
                y: [ty(selectedClusterMedoidPoint)],
                z: [tz(selectedClusterMedoidPoint)],
                hoverinfo: "skip",
                marker: {
                  size: 36.0 * selectedPulse,
                  color: selectedClusterColor,
                  opacity: 0.25,
                  line: { color: "rgba(255,255,255,0)", width: 0 },
                },
              },
              {
                type: "scatter3d",
                mode: "markers+text",
                name: "Selected cluster medoid",
                showlegend: false,
                x: [tx(selectedClusterMedoidPoint)],
                y: [ty(selectedClusterMedoidPoint)],
                z: [tz(selectedClusterMedoidPoint)],
                text: [medoidLabel],
                customdata: [selectedClusterMedoidPoint.player_key],
                hovertext: [`${getGalaxyHoverText(selectedClusterMedoidPoint)}<br>Cluster medoid`],
                hoverinfo: "none",
                textposition: "top center",
                textfont: {
                  family: "JetBrains Mono, monospace",
                  size: 19,
                  color: "rgba(255,255,255,0.99)",
                },
                marker: {
                  size: 17.6 * selectedPulse,
                  color: "#FFFFFF",
                  opacity: 1,
                  line: { color: hexToRgba(selectedClusterColor, 0.98), width: 2.65 },
                },
              },
            ];
          })()
        : [];

      const selectedClusterNameTrace = [];

      return [
        ...clusterConstellationTraces,
        ...similarityTrace,
        ...glowTraces,
        ...baseTraces,
        ...selectedClusterMedoidSunTrace,
        ...selectedNeighborGlowTrace,
        ...selectedGlowTrace,
        ...selectedNeighborTrace,
        ...selectedNeighborHitboxTrace,
        ...selectedTrace,
        ...selectedClusterNameTrace,
        ...hoverHitboxTraces,
      ];
    }

    const baseTraces = sortedEntries.map(([cluster, pts]) => {
      const clusterColor = getClusterColor(cluster);
      const isHighlighted = highlightedCluster === cluster;
      return {
        type: "scattergl",
        mode: showPlayerNames ? "markers+text" : "markers",
        name: `Cluster ${cluster}`,
        x: pts.map((p) => p.pc1),
        y: pts.map((p) => p.pc2),
        text: pts.map(formatPointSeasonName),
        hovertext: pts.map((p) => `${formatPointSeasonName(p)}<br>${p.season} · ${p.teams_played} · ${p.position}<br>Cluster ${p.cluster}`),
        customdata: pts.map((p) => p.player_key),
        hovertemplate: "%{hovertext}<extra></extra>",
        textposition: "top center",
        textfont: { family: "JetBrains Mono, monospace", size: 8.3, color: "rgba(223, 243, 244, 0.72)" },
        marker: {
          size: isHighlighted ? 9.65 : 8.75,
          color: pts.map((point) => getPointColor(cluster, point)),
          opacity: highlightedCluster == null ? 0.68 : isHighlighted ? 0.86 : 0.42,
          line: { color: isHighlighted ? hexToRgba(clusterColor, 0.56) : "rgba(223,243,244,0.11)", width: isHighlighted ? 0.9 : 0.72 },
        },
      };
    });

    const selectedTrace = selectedPoint
      ? [{
          type: "scattergl",
          mode: showPlayerNames ? "markers" : "markers+text",
          name: "Selected player",
          showlegend: false,
          x: [selectedPoint.pc1],
          y: [selectedPoint.pc2],
          text: [formatPointSeasonName(selectedPoint)],
          hovertext: [`${getGalaxyHoverText(selectedPoint)}<br>Selected player`],
          customdata: [selectedPoint.player_key],
          hovertemplate: "%{hovertext}<extra></extra>",
          textposition: "top center",
          textfont: { family: "JetBrains Mono, monospace", size: 9.8, color: "rgba(255, 255, 255, 0.96)" },
          marker: { size: 13.4, color: getSelectedPointColor(selectedPoint.cluster), opacity: 1, line: { color: "rgba(223, 243, 244, 0.76)", width: 1.8 } },
        }]
      : [];

    return [...baseTraces, ...selectedTrace];
  }, [clusterData, config, galaxyPlotEnabled, galaxyArchetypesEnabled, highlightedCluster, selectedPoint, showPlayerNames]);

  const galaxyArchetypeAnnotations = useMemo(() => {
    if (!clusterData || !galaxyPlotEnabled || !galaxyArchetypesEnabled) return [];

    const galaxy = clusterData.galaxy ?? {};
    const labels = Array.isArray(galaxy.archetype_labels) ? galaxy.archetype_labels : [];
    const visibleLabels = labels.filter((label) => (
      highlightedCluster != null ? Number(label.cluster) === Number(highlightedCluster) : true
    ));

    return visibleLabels.map((label) => {
      const clusterNumber = Number(label.cluster);
      const clusterColor = getClusterColor(clusterNumber);
      const clusterName = String(label.cluster_name ?? getConfigClusterName(config, clusterNumber, clusterData.algorithm, clusterData.distance_metric)).toUpperCase();
      const playerCount = Number(label.player_count ?? 0);

      return {
        x: Number(label.x ?? 0),
        y: Number(label.y ?? 0),
        z: Number(label.z ?? 0),
        text: `<b>${clusterName}</b><br><span style="font-size:10px">${Number.isFinite(playerCount) && playerCount > 0 ? playerCount : ""} PLAYERS</span>`,
        showarrow: false,
        xanchor: "center",
        yanchor: "middle",
        align: "center",
        opacity: 0.72,
        bgcolor: "rgba(2, 10, 13, 0.46)",
        bordercolor: hexToRgba(clusterColor, 0.38),
        borderpad: 6,
        borderwidth: 1,
        font: {
          family: "JetBrains Mono, monospace",
          size: highlightedCluster != null ? 17 : 13,
          color: hexToRgba(clusterColor, 0.98),
        },
      };
    });
  }, [clusterData, config, galaxyPlotEnabled, galaxyArchetypesEnabled, highlightedCluster]);

  const handleFeatureToggle = (feature) => {
    if (isFeatureLockedMode) return;

    setSelectedFeatures((prev) => {
      if (prev.includes(feature)) {
        if (prev.length === 1) return prev;
        return prev.filter((f) => f !== feature);
      }
      return [...prev, feature];
    });
  };

  const handleSelectorHoverStart = (feature, rect) => {
    if (isFeatureLockedMode) return;

    clearSelectorTooltipTimers();
    setHoveredSelectorFeature(feature);
    setHoveredSelectorRect(rect);
    setSelectorTooltipSize({ width: 0, height: 0 });
    setSelectorTooltipVisible(false);

    selectorShowTimerRef.current = window.setTimeout(() => {
      setHoveredSelectorFeature(feature);
      setHoveredSelectorRect(rect);
      setSelectorTooltipVisible(true);
    }, SELECTOR_TOOLTIP_SHOW_DELAY_MS);
  };

  const handleSelectorHoverMove = (feature, rect) => {
    setHoveredSelectorRect(rect);
    if (hoveredSelectorFeature !== feature) {
      setHoveredSelectorFeature(feature);
      setSelectorTooltipSize({ width: 0, height: 0 });
    }
  };

  const handleSelectorHoverEnd = () => {
    window.clearTimeout(selectorShowTimerRef.current);
    window.clearTimeout(selectorCleanupTimerRef.current);

    selectorHideTimerRef.current = window.setTimeout(() => {
      setSelectorTooltipVisible(false);
      selectorCleanupTimerRef.current = window.setTimeout(() => {
        setHoveredSelectorFeature(null);
        setHoveredSelectorRect(null);
        setSelectorTooltipSize({ width: 0, height: 0 });
      }, SELECTOR_TOOLTIP_EXIT_MS);
    }, SELECTOR_TOOLTIP_HIDE_DELAY_MS);
  };

  const openGlossary = () => {
    handleSelectorHoverEnd();
    clearGlossaryTimers();
    setGlossaryMounted(true);
    setGlossaryVisible(false);
    glossaryOpenTimerRef.current = window.setTimeout(() => {
      setGlossaryVisible(true);
    }, GLOSSARY_OPEN_DELAY_MS);
  };

  const closeGlossary = () => {
    clearGlossaryTimers();
    setGlossaryVisible(false);
    glossaryCleanupTimerRef.current = window.setTimeout(() => {
      setGlossaryMounted(false);
    }, GLOSSARY_EXIT_MS);
  };

  const clearSelectedGalaxyPlayerState = () => {
    setSelectedPoint(null);
    setSelectedDetail(null);
    setSelectedCareerMissingSeason(null);
    setSimilarPlayersSourcePoint(null);
    setSimilarPlayersData(null);
    setSimilarPlayersError("");
    setSkillBreakdownData(null);
    setSkillBreakdownError("");
    setThreePtBreakdownData(null);
    setThreePtBreakdownError("");
    setShowAllFeatures(false);
  };

  const handleGalaxyClusterDotClick = (clusterNumber) => {
    const numericCluster = Number(clusterNumber);
    if (!Number.isFinite(numericCluster)) return;
    galaxyInteractionActiveRef.current = false;
    window.clearTimeout(galaxyInteractionResumeTimerRef.current);

    if (galaxyArchetypesEnabled && Number(highlightedCluster) === numericCluster) {
      setGalaxyArchetypesEnabled(false);
      setHighlightedCluster(null);
      if (activeCenterView !== "plot") {
        transitionToCenterView("plot");
      }
      if (galaxyPlotEnabled) {
        window.requestAnimationFrame(() => resetGalaxyCameraFocus());
      }
      return;
    }

    clearSelectedGalaxyPlayerState();
    setHighlightedCluster(numericCluster);
    setGalaxyArchetypesEnabled(true);
    if (activeCenterView !== "plot") {
      transitionToCenterView("plot");
    }
    if (galaxyPlotEnabled) {
      window.requestAnimationFrame(() => focusGalaxyCameraOnCluster(numericCluster));
    }
  };

  const handleGalaxyArchetypeToggle = () => {
    setGalaxyArchetypesEnabled((previousValue) => {
      const nextEnabled = !previousValue;
      if (nextEnabled) {
        setHighlightedCluster(null);
      }
      return nextEnabled;
    });
  };

  const toggleHighlightedCluster = (clusterNumber) => {
    setHighlightedCluster((prev) => (prev === clusterNumber ? null : clusterNumber));
  };

  const getGalaxyClickCandidateKey = (candidate) => {
    const rawCustomData = candidate?.customdata;
    if (Array.isArray(rawCustomData)) return rawCustomData[0];
    return rawCustomData;
  };

  const rankGalaxyClickCandidate = (candidate) => {
    const traceName = String(candidate?.data?.name ?? "");
    if (traceName === "Selected player") return 0;
    if (traceName === "Closest connected comps") return 1;
    if (traceName === "Selected neighbor hitbox") return 2;
    if (traceName === "Selected cluster medoid") return 3;
    if (/^Cluster \d+$/.test(traceName)) return 4;
    if (traceName === "Galaxy player hitbox") return 5;
    return 9;
  };

  const resolveGalaxyClickTarget = (event) => {
    const candidates = Array.isArray(event?.points) ? event.points : [];
    const rankedCandidates = candidates
      .filter((candidate) => getGalaxyClickCandidateKey(candidate) != null)
      .filter((candidate) => !String(candidate?.data?.name ?? "").toLowerCase().includes("glow"))
      .sort((a, b) => rankGalaxyClickCandidate(a) - rankGalaxyClickCandidate(b));
    const winner = rankedCandidates[0];
    if (!winner) return null;
    const playerKey = getGalaxyClickCandidateKey(winner);
    const point = clusterData?.points?.find((p) => String(p.player_key) === String(playerKey)) ?? null;
    if (!point) return null;
    return {
      point,
      playerKey: String(playerKey),
      source: String(winner?.data?.name ?? "unknown"),
      priority: rankGalaxyClickCandidate(winner),
    };
  };

  const resolveGalaxyHoverCandidate = (event) => {
    if (!galaxyPlotEnabled || !clusterData?.points?.length) return null;
    const candidates = Array.isArray(event?.points) ? event.points : [];
    const rankedCandidates = candidates
      .filter((candidate) => getGalaxyClickCandidateKey(candidate) != null)
      .filter((candidate) => !String(candidate?.data?.name ?? "").toLowerCase().includes("glow"))
      .sort((a, b) => rankGalaxyClickCandidate(a) - rankGalaxyClickCandidate(b));
    const winner = rankedCandidates[0];
    if (!winner) return null;
    const playerKey = getGalaxyClickCandidateKey(winner);
    const point = clusterData.points.find((candidatePoint) => String(candidatePoint.player_key) === String(playerKey)) ?? null;
    if (!point) return null;
    return { point, plotlyPoint: winner };
  };

  const resolveGalaxyAnchorClientPosition = (event, plotlyPoint = null) => {
    const nativeEvent = event?.event ?? {};
    const pageX = Number(nativeEvent.pageX);
    const pageY = Number(nativeEvent.pageY);
    const directClientX = Number(nativeEvent.clientX ?? nativeEvent.x);
    const directClientY = Number(nativeEvent.clientY ?? nativeEvent.y);
    const scrollAdjustedPageX = Number.isFinite(pageX) ? pageX - window.scrollX : NaN;
    const scrollAdjustedPageY = Number.isFinite(pageY) ? pageY - window.scrollY : NaN;
    const directX = Number.isFinite(directClientX) ? directClientX : scrollAdjustedPageX;
    const directY = Number.isFinite(directClientY) ? directClientY : scrollAdjustedPageY;
    const nativePosition = Number.isFinite(directX) && Number.isFinite(directY) && directX >= 0 && directY >= 0
      ? { x: directX, y: directY }
      : null;

    const candidatePoint = plotlyPoint ?? (Array.isArray(event?.points) ? event.points[0] : null);
    const bbox = candidatePoint?.bbox ?? candidatePoint?.bb ?? null;
    const bboxX0 = Number(bbox?.x0 ?? bbox?.left);
    const bboxX1 = Number(bbox?.x1 ?? bbox?.right);
    const bboxY0 = Number(bbox?.y0 ?? bbox?.top);
    const bboxY1 = Number(bbox?.y1 ?? bbox?.bottom);
    const bboxCenterX = Number.isFinite(bboxX0) && Number.isFinite(bboxX1)
      ? (bboxX0 + bboxX1) / 2
      : NaN;
    const bboxCenterY = Number.isFinite(bboxY0) && Number.isFinite(bboxY1)
      ? (bboxY0 + bboxY1) / 2
      : NaN;

    if (Number.isFinite(bboxCenterX) && Number.isFinite(bboxCenterY) && bboxCenterX >= 0 && bboxCenterY >= 0) {
      const rawPosition = { x: bboxCenterX, y: bboxCenterY };
      const plotRect = plotWrapRef.current?.getBoundingClientRect?.();
      const shiftedPosition = plotRect
        ? { x: bboxCenterX + plotRect.left, y: bboxCenterY + plotRect.top }
        : null;

      if (nativePosition && shiftedPosition) {
        const rawDistance = Math.hypot(rawPosition.x - nativePosition.x, rawPosition.y - nativePosition.y);
        const shiftedDistance = Math.hypot(shiftedPosition.x - nativePosition.x, shiftedPosition.y - nativePosition.y);
        return shiftedDistance + 2 < rawDistance ? shiftedPosition : rawPosition;
      }
      if (plotRect) {
        const rawInsidePlot = rawPosition.x >= plotRect.left - 8
          && rawPosition.x <= plotRect.right + 8
          && rawPosition.y >= plotRect.top - 8
          && rawPosition.y <= plotRect.bottom + 8;
        const shiftedInsidePlot = shiftedPosition
          && shiftedPosition.x >= plotRect.left - 8
          && shiftedPosition.x <= plotRect.right + 8
          && shiftedPosition.y >= plotRect.top - 8
          && shiftedPosition.y <= plotRect.bottom + 8;
        if (!rawInsidePlot && shiftedInsidePlot) return shiftedPosition;
      }
      return rawPosition;
    }

    return nativePosition;
  };

  const getGalaxyHoverPlacement = (anchorPosition) => {
    if (!anchorPosition || typeof window === "undefined") return "right";
    const estimatedBubbleWidth = 138;
    const viewportPadding = 12;
    return anchorPosition.x + estimatedBubbleWidth + viewportPadding > window.innerWidth ? "left" : "right";
  };

  const buildGalaxyHoverTooltip = (point, anchorPosition, roleOverride = null) => {
    const clusterName = getConfigClusterName(
      config,
      point.cluster,
      clusterData?.algorithm,
      clusterData?.distance_metric
    );
    return {
      key: String(point.player_key),
      x: anchorPosition.x,
      y: anchorPosition.y,
      placement: getGalaxyHoverPlacement(anchorPosition),
      title: formatPointSeasonName(point),
      meta: `${point.season} · ${point.teams_played} · ${point.position}`,
      archetype: clusterName,
      accentColor: getClusterColor(point.cluster),
      role: roleOverride ?? (selectedPoint?.player_key === point.player_key ? "SELECTED PLAYER" : "PLAYER-SEASON"),
    };
  };

  const handleGalaxyPlotHover = (event) => {
    if (!galaxyPlotEnabled) return;
    const hoverCandidate = resolveGalaxyHoverCandidate(event);
    const anchorPosition = resolveGalaxyAnchorClientPosition(event, hoverCandidate?.plotlyPoint);
    if (!hoverCandidate?.point || !anchorPosition) {
      setGalaxyHoverTooltip(null);
      return;
    }
    setGalaxyHoverTooltip(buildGalaxyHoverTooltip(hoverCandidate.point, anchorPosition));
  };

  const handleGalaxyPlotUnhover = () => {
    setGalaxyHoverTooltip(null);
  };

  const selectGalaxyPointFromPlotClick = (event) => {
    const target = resolveGalaxyClickTarget(event);
    if (!target?.point) return;
    const clickPoint = Array.isArray(event?.points)
      ? event.points.find((candidate) => String(getGalaxyClickCandidateKey(candidate)) === String(target.playerKey))
      : null;
    const anchorPosition = resolveGalaxyAnchorClientPosition(event, clickPoint);
    if (anchorPosition) {
      setGalaxyHoverTooltip(buildGalaxyHoverTooltip(target.point, anchorPosition, "PLAYER-SEASON"));
    }
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const lastSelection = galaxyLastClickSelectionRef.current;
    if (
      lastSelection?.playerKey
      && target.playerKey !== lastSelection.playerKey
      && now - Number(lastSelection.timestamp ?? 0) < GALAXY_CLICK_SELECTION_LOCK_MS
    ) {
      return;
    }
    galaxyLastClickSelectionRef.current = { playerKey: target.playerKey, timestamp: now };
    selectGalaxyPoint(target.point);
  };

  const handleAlgorithmChange = (algorithm) => {
    if (algorithm !== "kmeans") return;
    setSelectedAlgorithm("kmeans");
    setAlgorithmMenuOpen(false);
    setVisualizationMenuOpen(false);
    setClusterCounts((prev) => ({
      ...prev,
      kmeans: prev.kmeans ?? (config?.euclidean_kmeans_locked_k ?? config?.default_kmeans_k ?? config?.default_k ?? 12),
    }));
    setHighlightedCluster(null);
    setGalaxyArchetypesEnabled(false);
  };

  const handleDistanceMetricChange = (metric) => {
    if (metric !== "euclidean") return;
    clearSelectorTooltipTimers();
    setHoveredSelectorFeature(null);
    setSelectorTooltipVisible(false);
    setSelectedDistanceMetric("euclidean");
    setDistanceMetricMenuOpen(false);
    setVisualizationMenuOpen(false);
    setHighlightedCluster(null);
    setGalaxyArchetypesEnabled(false);
  };

  const handleVisualizationModeChange = (mode) => {
    if (mode === "3d_galaxy" && !galaxyDataAvailable) return;
    clearViewTransitionTimers();
    setViewTransitionActive(false);
    setActiveCenterView("plot");
    setSelectedVisualizationMode(mode);
    setVisualizationMenuOpen(false);
    setAlgorithmMenuOpen(false);
    setDistanceMetricMenuOpen(false);
  };

  const selectorTooltipStyle = useMemo(() => {
    if (!hoveredSelectorRect) return null;
    if (typeof window === "undefined") return null;

    const viewportPadding = 14;
    const tooltipGap = 14;
    const fallbackWidth = Math.min(420, Math.max(280, window.innerWidth * 0.26));
    const fallbackHeight = 176;
    const tooltipWidth = selectorTooltipSize.width || fallbackWidth;
    const tooltipHeight = selectorTooltipSize.height || fallbackHeight;

    let left = hoveredSelectorRect.right + tooltipGap;
    if (left + tooltipWidth + viewportPadding > window.innerWidth) {
      left = Math.max(
        viewportPadding,
        hoveredSelectorRect.left - tooltipWidth - tooltipGap
      );
    }

    const centeredTop =
      hoveredSelectorRect.top + hoveredSelectorRect.height / 2 - tooltipHeight / 2;
    const top = Math.max(
      viewportPadding,
      Math.min(centeredTop, window.innerHeight - tooltipHeight - viewportPadding)
    );

    return {
      left: Math.round(left),
      top: Math.round(top),
      maxWidth: `min(420px, calc(100vw - ${Math.round(left) + viewportPadding}px))`,
      maxHeight: `calc(100vh - ${Math.round(top) + viewportPadding}px)`,
    };
  }, [hoveredSelectorRect, selectorTooltipSize]);

  const totalGlossaryMetrics = useMemo(
    () => GLOSSARY_SECTIONS.reduce((sum, section) => sum + section.features.length, 0),
    []
  );

  const renderPlayerSearchControl = ({ compact = false } = {}) => (
    <div
      className={`plot-control-card player-search-card ${compact ? "galaxy-fullscreen-player-search" : ""}`.trim()}
      ref={playerSearchRef}
      onMouseDownCapture={(event) => event.stopPropagation()}
      onPointerDownCapture={(event) => event.stopPropagation()}
      onClick={(event) => event.stopPropagation()}
    >
      {!compact && <span className="plot-control-label">PLAYER SEARCH</span>}
      <div className="player-search-shell">
        <span className="player-search-icon">
          <SearchIcon />
        </span>
        <input
          ref={playerSearchInputRef}
          className="player-search-input"
          value={playerSearch}
          onChange={handlePlayerSearchChange}
          onFocus={() => {
            if (playerSearch.trim()) {
              setPlayerSearchOpen(true);
            }
          }}
          onKeyDown={handlePlayerSearchKeyDown}
          placeholder={compact ? "search player..." : "search player + season..."}
          autoComplete="off"
        />
      </div>

      {playerSearchOpen && playerSearch.trim() && (
        <div className="player-search-results">
          {playerSearchResults.length > 0 ? (
            playerSearchResults.map((entry, index) => (
              <button
                key={entry.point.player_key}
                type="button"
                className={`player-search-result ${index === activeSearchIndex ? "active" : ""}`}
                onMouseDown={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                }}
                onClick={(event) => {
                  event.stopPropagation();
                  handlePlayerSearchSelect(entry.point);
                }}
                onMouseEnter={() => setActiveSearchIndex(index)}
              >
                <SearchClusterBadge point={entry.point} algorithm={currentAlgorithm} />
                <span className="player-search-result-copy">
                  <span className="player-search-result-name">{entry.point.player_name}</span>
                  <span className="player-search-result-meta">
                    {entry.point.season} · {entry.point.teams_played} · {entry.point.position}
                  </span>
                </span>
              </button>
            ))
          ) : (
            <div className="player-search-empty">NO_MATCHES_FOUND</div>
          )}
        </div>
      )}
    </div>
  );

  const renderSelectedPlayerProfileHeader = ({ compact = false } = {}) => {
    if (!selectedPoint) return null;

    return (
      <div className={`player-report-header-card ${compact ? "galaxy-compact-profile-header" : ""}`.trim()}>
        <div className="player-report-identity-row">
          <PlayerHeadshot
            src={selectedDetail?.meta?.headshot_url || selectedPoint.headshot_url}
            name={selectedPoint.player_name}
            size="large"
          />

          <div className="player-report-identity-copy">
            <div className="player-headline">
              <div className="player-name">{selectedPoint.player_name}</div>
              <div className="player-meta">
                {selectedPoint.season} · {selectedPoint.teams_played} · {selectedPoint.position}
              </div>
            </div>

            {isFuzzyMode && (
              <ProbabilityBar memberships={selectedPoint.memberships ?? []} />
            )}
          </div>
        </div>

        {!isFuzzyMode && (
          <div className="player-report-cluster-label player-report-archetype-full">
            {getConfigClusterName(config, selectedPoint.cluster, currentAlgorithm, selectedDistanceMetric)}
          </div>
        )}

        {!isFuzzyMode && (
          <PlayerBadges badges={selectedDetail?.badges ?? []} />
        )}

        {!isFuzzyMode && (
          compact ? (
            <div className="player-report-action-row galaxy-compact-action-row">
              <button
                type="button"
                className="chip accent career-path-chip compact player-report-action-btn"
                onClick={handleOpenCareerPath}
              >
                Career Path
              </button>
              <button
                type="button"
                className="show-all-btn similar-players-drawer-btn player-report-similar-btn player-report-action-btn"
                onClick={handleOpenSimilarPlayers}
              >
                Similar Players
              </button>
              <button
                type="button"
                className="show-all-btn player-report-skill-btn player-report-action-btn"
                onClick={handleOpenSkillBreakdown}
              >
                Skill Breakdown
              </button>
            </div>
          ) : (
            <>
              <div className="player-report-action-row">
                <button
                  type="button"
                  className="chip accent career-path-chip compact player-report-action-btn"
                  onClick={handleOpenCareerPath}
                >
                  Career Path
                </button>
                <button
                  type="button"
                  className="show-all-btn similar-players-drawer-btn player-report-similar-btn player-report-action-btn"
                  onClick={handleOpenSimilarPlayers}
                >
                  Similar Players
                </button>
              </div>

              <div className="player-report-action-row-secondary">
                <button
                  type="button"
                  className="show-all-btn player-report-skill-btn"
                  onClick={handleOpenSkillBreakdown}
                >
                  Skill Breakdown
                </button>
              </div>
            </>
          )
        )}
      </div>
    );
  };

  const renderGalaxyBestWorstPanel = () => {
    if (!selectedPoint) return null;

    return (
      <div className="galaxy-best-worst-panel">
        {loadingDetail && <div className="empty-state">Loading player detail...</div>}

        {!loadingDetail && selectedDetail && (
          <>
            <div className="report-title-row galaxy-best-worst-title-row">
              <div className="report-subtitle report-subtitle-tight">TOP_PERCENTILE_FEATURES</div>
              <button
                type="button"
                className="show-all-btn galaxy-show-all-btn"
                onClick={() => setShowAllFeatures(true)}
              >
                SHOW_ALL
              </button>
            </div>

            <div className="feature-summary-list galaxy-feature-summary-list">
              {selectedDetail.top_features.map((item) => (
                <FeatureStatCard
                  key={`galaxy-top-${item.feature}`}
                  feature={item.feature}
                  value={item.value}
                  percentile={item.percentile}
                  label={item.label}
                />
              ))}
            </div>

            <div className="report-subtitle galaxy-low-subtitle">LOW_PERCENTILE_FEATURES</div>
            <div className="feature-summary-list galaxy-feature-summary-list">
              {selectedDetail.bottom_features.map((item) => (
                <FeatureStatCard
                  key={`galaxy-bottom-${item.feature}`}
                  feature={item.feature}
                  value={item.value}
                  percentile={item.percentile}
                  label={item.label}
                />
              ))}
            </div>
          </>
        )}
      </div>
    );
  };

  const renderGalaxyPlayerProfileOverlay = () => {
    if (!galaxyFullscreenPlotActive || selectedCareerMissingSeason) return null;
    if (highlightedCluster != null && !selectedPoint) return null;

    if (selectedPoint && galaxyPlayerProfileHidden) {
      return (
        <button
          type="button"
          className="galaxy-show-profile-btn"
          onClick={(event) => {
            event.stopPropagation();
            setGalaxyPlayerProfileHidden(false);
          }}
          onMouseDown={(event) => event.stopPropagation()}
          onPointerDown={(event) => event.stopPropagation()}
        >
          SHOW PLAYER PROFILE
        </button>
      );
    }

    return (
      <aside
        className="galaxy-player-profile-card neon-panel"
        onMouseDownCapture={(event) => event.stopPropagation()}
        onPointerDownCapture={(event) => event.stopPropagation()}
        onWheelCapture={(event) => event.stopPropagation()}
      >
        <div className="galaxy-profile-topline">
          <div className="section-header">// PLAYER_REPORT</div>
          {selectedPoint && (
            <button
              type="button"
              className="galaxy-profile-hide-btn"
              onClick={() => {
                setGalaxyPlayerProfileHidden(true);
                setGalaxyBestWorstOpen(false);
                setShowAllFeatures(false);
              }}
            >
              HIDE PLAYER PROFILE
            </button>
          )}
        </div>

        {!selectedPoint && (
          <div className="empty-state">
            Click any player point to view player report.
          </div>
        )}

        {selectedPoint && renderSelectedPlayerProfileHeader({ compact: true })}

        {selectedPoint && !isFuzzyMode && (
          <div className="player-report-action-row-secondary galaxy-best-worst-toggle-row">
            <button
              type="button"
              className={`show-all-btn player-report-skill-btn galaxy-best-worst-toggle ${galaxyBestWorstOpen ? "active" : ""}`}
              onClick={() => {
                const nextGalaxyBestWorstOpen = !galaxyBestWorstOpen;
                setGalaxyBestWorstOpen(nextGalaxyBestWorstOpen);
                if (!nextGalaxyBestWorstOpen) setShowAllFeatures(false);
              }}
            >
              {galaxyBestWorstOpen ? "Hide Best/Worst Features" : "Show Best/Worst Features"}
            </button>
          </div>
        )}

        {selectedPoint && galaxyBestWorstOpen && renderGalaxyBestWorstPanel()}
      </aside>
    );
  };

  const renderGalaxyAllFeaturesOverlay = () => {
    if (!galaxyFullscreenPlotActive || !showAllFeatures || !selectedDetail) return null;

    return (
      <aside
        className="galaxy-all-features-panel neon-panel"
        onMouseDownCapture={(event) => event.stopPropagation()}
        onPointerDownCapture={(event) => event.stopPropagation()}
        onWheelCapture={(event) => event.stopPropagation()}
      >
        <div className="report-title-row drawer-header-row galaxy-all-features-header">
          <div>
            <div className="section-header">// ALL_FEATURES</div>
            <div className="drawer-player-line">
              {selectedDetail?.meta?.player_name} · {selectedDetail?.meta?.season} · {selectedDetail?.meta?.teams_played}
            </div>
          </div>
          <button className="show-all-btn" onClick={() => setShowAllFeatures(false)}>
            CLOSE
          </button>
        </div>

        <div className="all-features-list galaxy-all-features-list">
          {selectedDetail?.stats.map((item) => (
            <FeatureStatCard
              key={`galaxy-all-${item.feature}`}
              feature={item.feature}
              value={item.value}
              percentile={item.percentile}
              label={item.label}
            />
          ))}
        </div>
      </aside>
    );
  };

  return (
    <div className="app-shell">
      <WelcomeModal />
      <div className="grid-bg" />
      {startupLoaderVisible && (
        <div className="startup-loading-screen" aria-live="polite" role="status">
          <div className="startup-loading-card neon-panel">Galaxy Loading...</div>
        </div>
      )}
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◎</span>
          <span>Basketball and Machine Learning</span>
        </div>
        <nav className="nav-strip">
          <span>[STATS]</span>
          <span>[GALAXY]</span>
          <span>[CLUSTER_LAB]</span>
        </nav>
      </header>

      {hoveredSelectorFeature && selectorTooltipStyle && (
        <FeatureTooltip
          ref={selectorTooltipRef}
          feature={hoveredSelectorFeature}
          className="feature-tooltip-fixed"
          style={selectorTooltipStyle}
          visible={selectorTooltipVisible}
        />
      )}

      {glossaryMounted && (
        <div
          className={`glossary-overlay ${glossaryVisible ? "visible" : ""}`}
          onClick={closeGlossary}
        >
          <div
            className={`glossary-modal neon-panel ${glossaryVisible ? "visible" : ""}`}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="glossary-modal-header">
              <div>
                <div className="section-header glossary-title-row">
                  <span>▣ STAT_GLOSSARY</span>
                  <span className="glossary-title-count">[{totalGlossaryMetrics} METRICS]</span>
                </div>
              </div>
              <button className="glossary-close-btn" onClick={closeGlossary}>
                ×
              </button>
            </div>

            <div className="glossary-modal-body">
              {GLOSSARY_SECTIONS.map((section) => (
                <GlossarySection
                  key={section.key}
                  section={section}
                  open={Boolean(glossarySectionsOpen[section.key])}
                  onToggle={() =>
                    setGlossarySectionsOpen((prev) => ({
                      ...prev,
                      [section.key]: !prev[section.key],
                    }))
                  }
                />
              ))}
            </div>

            <div className="glossary-modal-footer">PRESS ESC OR CLICK OUTSIDE TO CLOSE</div>
          </div>
        </div>
      )}

      <MethodologyModal
        open={methodologyOpen}
        onClose={() => setMethodologyOpen(false)}
        totalGlossaryMetrics={totalGlossaryMetrics}
      />

      <ReadMeModal
        open={readMeOpen}
        onClose={() => setReadMeOpen(false)}
      />

      <main
        ref={mainLayoutRef}
        className={`main-layout ${drawerOpen ? "drawer-open" : ""} ${galaxyFullscreenEnabled && galaxyPlotEnabled ? "galaxy-fullscreen" : ""}`}
        style={{
          "--left-panel-width": `${leftPanelWidth}px`,
          "--right-panel-width": `${rightPanelWidth}px`,
          "--drawer-panel-width": `${DEFAULT_DRAWER_PANEL_WIDTH}px`,
          "--resize-gutter-width": `${RESIZE_GUTTER_WIDTH}px`,
        }}
      >
        <aside className={`left-panel neon-panel ${leftPanelWidth <= COLLAPSED_PANEL_THRESHOLD ? "collapsed-panel" : ""}`}>
          <div className="section-header">// CONTROL_PANEL</div>

          <div className="control-block">
            <label className="control-label">DATASET</label>
            <div className="value-box small">{config?.dataset_path ?? "loading..."}</div>
          </div>

          <div className="control-block">
            <label className="control-label">{getClusterControlLabel(selectedAlgorithm)}</label>
            <div className={`k-row-shell ${isClusterCountLockedMode ? "locked" : ""}`}>
              {isClusterCountLockedMode && <FeatureLockOverlay />}
              <div className="k-row">
                <input
                  className={`k-slider ${isClusterCountLockedMode ? "locked" : ""}`}
                  type="range"
                  min={2}
                  max={20}
                  value={activeClusterCount}
                  disabled={isClusterCountLockedMode}
                  onChange={(e) => {
                    if (isClusterCountLockedMode) return;
                    setClusterCounts((prev) => ({
                      ...prev,
                      [selectedAlgorithm]: Number(e.target.value),
                    }));
                  }}
                />
                <div className={`value-box tight ${isClusterCountLockedMode ? "locked" : ""}`}>{activeClusterCount}</div>
              </div>
            </div>
          </div>

          <div className="control-block">
            <label className="control-label">FEATURE_SEARCH</label>
            <input
              className={`search-input ${isFeatureLockedMode ? "locked" : ""}`}
              value={featureFilter}
              onChange={(e) => setFeatureFilter(e.target.value)}
              placeholder={isFeatureLockedMode ? "search locked preset..." : "search feature..."}
            />
          </div>

          <div className="feature-toolbar">
            <button
              className="mini-btn"
              onClick={() => config && setSelectedFeatures(config.allowed_features)}
              disabled={isFeatureLockedMode}
            >
              SELECT_ALL
            </button>
            <button
              className="mini-btn"
              onClick={() => config && setSelectedFeatures(config.default_features ?? config.allowed_features)}
              disabled={isFeatureLockedMode}
            >
              DEFAULT
            </button>
          </div>

          <div className={`feature-count ${isFeatureLockedMode ? "locked" : ""}`}>
            {isFeatureLockedMode ? `LOCKED PRESET · ${lockedFeatureList.length} FEATURES` : `${selectedFeatures.length} selected`}
          </div>

          <div className={`feature-list-shell ${isFeatureLockedMode ? "locked" : ""}`}>
            {isFeatureLockedMode && <FeatureLockOverlay />}
            <div className={`feature-list ${isFeatureLockedMode ? "feature-list-locked" : ""}`}>
              {visibleFeatures.map((feature) => {
                const active = displayedSelectedFeatures.includes(feature);
                return (
                  <FeatureSelectorButton
                    key={feature}
                    feature={feature}
                    active={active}
                    disabled={isFeatureLockedMode}
                    onClick={() => handleFeatureToggle(feature)}
                    onHoverStart={handleSelectorHoverStart}
                    onHoverMove={handleSelectorHoverMove}
                    onHoverEnd={handleSelectorHoverEnd}
                  />
                );
              })}
            </div>
          </div>
        </aside>

        <div
          className="resize-gutter left-resize-gutter"
          onMouseDown={(event) => startPanelResize("left", event)}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize control panel"
        >
          <span className="resize-gutter-handle" />
        </div>

        <section className="center-panel">
          <div className="plot-header neon-panel">
            <div
              className="plot-control-strip"
              onMouseDownCapture={(event) => event.stopPropagation()}
              onPointerDownCapture={(event) => event.stopPropagation()}
            >
              <button className="plot-control-card plot-control-button" onClick={openGlossary}>
                <span className="plot-control-label">GLOSSARY</span>
                <span className="plot-control-value">{totalGlossaryMetrics} metrics</span>
              </button>

              <div className="plot-control-card plot-control-card-select" ref={algorithmMenuRef}>
                <button
                  type="button"
                  className="plot-control-select-btn"
                  onClick={() => {
                    setAlgorithmMenuOpen((prev) => !prev);
                    setDistanceMetricMenuOpen(false);
                    setVisualizationMenuOpen(false);
                  }}
                >
                  <span className="plot-control-label">CLUSTERING ALGORITHM</span>
                  <span className="plot-control-value">{getAlgorithmLabel(selectedAlgorithm)}</span>
                </button>

                {algorithmMenuOpen && (
                  <div className="algorithm-menu">
                    {["kmeans"].map((algorithm) => (
                      <button
                        key={algorithm}
                        type="button"
                        className={`algorithm-menu-item ${selectedAlgorithm === algorithm ? "active" : ""}`}
                        onClick={() => {
                            handleAlgorithmChange(algorithm);
                            setDistanceMetricMenuOpen(false);
                          }}
                      >
                        {getAlgorithmLabel(algorithm)}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="plot-control-card plot-control-card-select" ref={visualizationMenuRef}>
                <button
                  type="button"
                  className="plot-control-select-btn"
                  onClick={() => {
                    setVisualizationMenuOpen((prev) => !prev);
                    setAlgorithmMenuOpen(false);
                    setDistanceMetricMenuOpen(false);
                  }}
                >
                  <span className="plot-control-label">VISUALIZATION</span>
                  <span className="plot-control-value">{activeVisualizationLabel}</span>
                </button>

                {visualizationMenuOpen && (
                  <div className="algorithm-menu">
                    {VISUALIZATION_MODE_OPTIONS.map((mode) => {
                      const disabled = mode === "3d_galaxy" && !galaxyDataAvailable;
                      return (
                        <button
                          key={mode}
                          type="button"
                          className={`algorithm-menu-item ${selectedVisualizationMode === mode ? "active" : ""}`}
                          onClick={() => handleVisualizationModeChange(mode)}
                          disabled={disabled}
                          title={disabled ? "3D galaxy assets are not available for the active dataset." : undefined}
                        >
                          {getVisualizationModeLabel(mode)}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="plot-control-card plot-control-card-select" ref={distanceMetricMenuRef}>
                <button
                  type="button"
                  className="plot-control-select-btn"
                  onClick={() => {
                    setDistanceMetricMenuOpen((prev) => !prev);
                    setAlgorithmMenuOpen(false);
                    setVisualizationMenuOpen(false);
                  }}
                >
                  <span className="plot-control-label">DISTANCE METRIC</span>
                  <span className="plot-control-value">{getDistanceMetricLabel(selectedDistanceMetric)}</span>
                </button>

                {distanceMetricMenuOpen && (
                  <div className="algorithm-menu">
                    {DISTANCE_METRIC_OPTIONS.map((metric) => (
                      <button
                        key={metric}
                        type="button"
                        className={`algorithm-menu-item ${selectedDistanceMetric === metric ? "active" : ""}`}
                        onClick={() => handleDistanceMetricChange(metric)}
                      >
                        {getDistanceMetricLabel(metric)}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {renderPlayerSearchControl()}

              <button
                type="button"
                className="plot-control-card plot-control-button methodology-control-card"
                onClick={(event) => {
                  event.stopPropagation();
                  setAlgorithmMenuOpen(false);
                  setVisualizationMenuOpen(false);
                  setDistanceMetricMenuOpen(false);
                  setPlayerSearchOpen(false);
                  setMethodologyOpen(true);
                }}
              >
                <span className="plot-control-label">INFO</span>
                <span className="plot-control-value">Galaxy</span>
              </button>
            </div>

            {highlightedCluster != null && (
              <div className={`header-side-strip ${nonScatterViewEnabled ? "with-return-action" : ""}`}>
                {nonScatterViewEnabled && (
                  <button
                    type="button"
                    className={`plot-control-card plot-control-button return-scatter-card ${viewTransitionActive ? "transitioning" : ""}`}
                    onClick={handleBackToGalaxy}
                    title="Return to galaxy view"
                  >
                    <span className="plot-control-label">RETURN TO</span>
                    <span className="plot-control-value">GALAXY</span>
                  </button>
                )}

                <button
                  type="button"
                  className={`plot-control-card plot-control-button cluster-description-toggle-card ${viewTransitionActive ? "transitioning" : ""}`}
                  onClick={handleOpenClusterDescription}
                  title={highlightedClusterName}
                >
                  <span className="plot-control-label">VIEW CLUSTER DESCRIPTION</span>
                  <span className="plot-control-value cluster-description-control-name">{highlightedClusterName}</span>
                </button>
              </div>
            )}
          </div>

          <div className="plot-shell neon-panel">
            {loadingClusters && <div className="loading-overlay">{galaxyPlotEnabled ? "LOADING_GALAXY..." : "LOADING_SCATTER..."}</div>}
            {!loadingClusters && error && <div className="error-box">{error}</div>}

            <div className={`plot-chart-stage ${clusterDescriptionViewEnabled ? "cluster-description-active" : ""} ${careerPathViewEnabled ? "career-path-active" : ""} ${similarPlayersViewEnabled ? "similar-players-active" : ""} ${skillBreakdownViewEnabled ? "skill-breakdown-active" : ""} ${threePtBreakdownViewEnabled ? "three-pt-breakdown-active" : ""}`}>
              {clusterDescriptionViewEnabled ? (
                <ClusterDescriptionView
                  report={selectedClusterReport}
                  clusterColor={getClusterColor(highlightedCluster ?? 1)}
                  onBack={handleBackToGalaxy}
                  loading={loadingClusterReport}
                  error={clusterReportError}
                />
              ) : similarPlayersViewEnabled ? (
                <SimilarPlayersView
                  data={similarPlayersData}
                  sourcePoint={similarPlayersSourcePoint}
                  loading={loadingSimilarPlayers}
                  error={similarPlayersError}
                  onBack={handleBackToGalaxy}
                  onSelectSimilarPlayer={handleSelectSimilarPlayer}
                />
              ) : skillBreakdownViewEnabled ? (
                <SkillBreakdownView
                  data={skillBreakdownData}
                  sourcePoint={selectedPoint}
                  loading={loadingSkillBreakdown}
                  error={skillBreakdownError}
                  onBack={handleBackToGalaxy}
                  onOpenThreePtBreakdown={handleOpenThreePtBreakdown}
                />
              ) : threePtBreakdownViewEnabled ? (
                <ThreePtBreakdownView
                  data={threePtBreakdownData}
                  sourcePoint={selectedPoint}
                  loading={loadingThreePtBreakdown}
                  error={threePtBreakdownError}
                  onBack={handleBackToGalaxy}
                />
              ) : careerPathViewEnabled ? (
                <CareerPathView
                  playerName={careerPathPlayerName}
                  timeline={careerPathTimeline}
                  clusterItems={clusterLegendItems}
                  selectedPointKey={selectedPoint?.player_key}
                  selectedMissingSeason={selectedCareerMissingSeason?.season ?? null}
                  onBack={handleBackToGalaxy}
                  onSelectQualifiedPoint={(point) => {
                    if (!point) return;
                    selectGalaxyPoint(point, { updateCareerPath: true });
                  }}
                  onSelectMissingSeason={(node) => {
                    setSelectedPoint(null);
                    setSelectedCareerMissingSeason({
                      player_name: careerPathPlayerName,
                      season: node.season,
                    });
                    setShowAllFeatures(false);
                  }}
                />
              ) : (
                <>
                  {clusterLegendItems.length > 0 && (
                    <div className="cluster-legend-row neon-panel">
                      {!galaxyPlotEnabled && (
                        <button
                          type="button"
                          className={`player-name-toggle ${showPlayerNames ? "active" : ""}`}
                          onClick={() => setShowPlayerNames((previousValue) => !previousValue)}
                          aria-pressed={showPlayerNames}
                        >
                          <span className="player-name-toggle-label">SHOW PLAYER NAMES</span>
                          <span className="player-name-toggle-value">{showPlayerNames ? "ON" : "OFF"}</span>
                        </button>
                      )}

                      {galaxyPlotEnabled && (
                        <button
                          type="button"
                          className={`player-name-toggle archetype-constellation-toggle ${galaxyArchetypesEnabled ? "active" : ""}`}
                          onClick={handleGalaxyArchetypeToggle}
                          aria-pressed={galaxyArchetypesEnabled}
                          title="Connect same-archetype players with truth-space constellation edges"
                        >
                          <span className="player-name-toggle-label">ARCHETYPES</span>
                          <span className="player-name-toggle-value">{galaxyArchetypesEnabled ? "ON" : "OFF"}</span>
                        </button>
                      )}

                      {galaxyPlotEnabled && galaxyFullscreenEnabled && renderPlayerSearchControl({ compact: true })}

                      {clusterLegendItems.map((item) => (
                        <button
                          key={`legend-${item.cluster}`}
                          type="button"
                          className={`cluster-legend-btn cluster-legend-dot-btn ${highlightedCluster === item.cluster ? "highlighted" : ""}`}
                          onClick={() => handleGalaxyClusterDotClick(item.cluster)}
                          style={{
                            "--cluster-color": item.color,
                          }}
                          title={`${item.name} — click to view this archetype`}
                          aria-label={`View ${item.name} archetype`}
                        >
                          <span className="cluster-legend-dot" />
                        </button>
                      ))}

                      {galaxyPlotEnabled && (
                        <>
                          <span className="cluster-legend-hint">click colored dots to view archetypes</span>
                        </>
                      )}

                      {galaxyPlotEnabled && (
                        <>
                          <button
                            type="button"
                            className="player-name-toggle browser-fullscreen-btn"
                            onClick={handleBrowserFullscreenToggle}
                            title="Toggle true browser fullscreen"
                          >
                            <span className="player-name-toggle-label">{browserFullscreenActive ? "EXIT FULL SCREEN" : "ENTER FULL SCREEN"}</span>
                          </button>
                          <span className="cluster-legend-future-note cluster-legend-future-note-after-fullscreen">all positions · 16 archetypes</span>
                        </>
                      )}

                    </div>
                  )}

                  <div
                    className={`plot-chart-wrap ${galaxyPlotEnabled ? "galaxy-chart-wrap" : ""} ${galaxyPlotEnabled && selectedPoint ? "galaxy-focused" : ""}`}
                    ref={plotWrapRef}
                    onDoubleClick={() => {
                      if (!galaxyPlotEnabled) {
                        setPlotAxisRange(null);
                      }
                    }}
                  >

                    {galaxyPlotEnabled && galaxyHoverTooltip && (
                      <div
                        className={`galaxy-player-hover-anchor galaxy-player-hover-anchor--${galaxyHoverTooltip.placement ?? "right"}`}
                        style={{
                          left: `${Math.round(galaxyHoverTooltip.x)}px`,
                          top: `${Math.round(galaxyHoverTooltip.y)}px`,
                          "--tooltip-accent": galaxyHoverTooltip.accentColor,
                        }}
                      >
                        <div className="galaxy-player-hover-bubble">
                          <div className="galaxy-player-hover-kicker">{galaxyHoverTooltip.role}</div>
                          <div className="galaxy-player-hover-title">{galaxyHoverTooltip.title}</div>
                          <div className="galaxy-player-hover-meta">{galaxyHoverTooltip.meta}</div>
                          <div className="galaxy-player-hover-archetype">{galaxyHoverTooltip.archetype}</div>
                        </div>
                      </div>
                    )}

                    {galaxyFullscreenPlotActive && highlightedCluster != null && (
                      <button
                        type="button"
                        className={`galaxy-canvas-cluster-description-btn ${selectedPoint && !galaxyPlayerProfileHidden ? "with-profile" : ""} ${viewTransitionActive ? "transitioning" : ""}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          handleOpenClusterDescription();
                        }}
                        title={highlightedClusterName}
                      >
                        View Cluster Description
                      </button>
                    )}

                    {galaxyFullscreenPlotActive && (
                      <>
                        <button
                          type="button"
                          className="readme-floating-btn"
                          onClick={(event) => {
                            event.stopPropagation();
                            setReadMeOpen(true);
                          }}
                          onMouseDown={(event) => event.stopPropagation()}
                          onPointerDown={(event) => event.stopPropagation()}
                          title="Open site read me"
                        >
                          Read Me
                        </button>

                        <div
                          className="universe-info-floating-stack"
                          onMouseDownCapture={(event) => event.stopPropagation()}
                          onPointerDownCapture={(event) => event.stopPropagation()}
                          onWheelCapture={(event) => event.stopPropagation()}
                        >
                          <button
                            type="button"
                            className="universe-info-floating-btn"
                            onClick={(event) => {
                              event.stopPropagation();
                              setMethodologyOpen(true);
                            }}
                            title="Open Galaxy explanation"
                          >
                            What is the Galaxy?
                          </button>
                          <a
                            className="galaxy-contact-link"
                            href="mailto:harsha9anand@gmail.com"
                            onClick={(event) => event.stopPropagation()}
                          >
                            <svg className="galaxy-contact-mail-icon" viewBox="0 0 24 24" aria-hidden="true">
                              <path d="M4 6h16v12H4V6Z" />
                              <path d="m4 7 8 6 8-6" />
                            </svg>
                            <span>harsha9anand@gmail.com</span>
                          </a>
                        </div>
                      </>
                    )}

                    <Plot
                      data={traces}
                      layout={{
                        autosize: true,
                        showlegend: false,
                        dragmode: galaxyPlotEnabled ? "orbit" : false,
                        paper_bgcolor: "rgba(0,0,0,0)",
                        plot_bgcolor: "rgba(0,0,0,0)",
                        margin: galaxyPlotEnabled ? { l: 0, r: 0, b: 0, t: 0 } : PLOT_LAYOUT_MARGIN,
                        font: { family: "JetBrains Mono, monospace", color: "#DFF3F4" },
                        uirevision: galaxyPlotEnabled ? "galaxy-camera-state" : "pca-2d-state",
                        hovermode: "closest",
                        hoverdistance: galaxyPlotEnabled ? 42 : 18,
                        spikedistance: -1,
                        ...(galaxyPlotEnabled
                          ? {
                              scene: {
                                bgcolor: "rgba(0,0,0,0)",
                                aspectmode: "cube",
                                xaxis: { visible: false, showspikes: false },
                                yaxis: { visible: false, showspikes: false },
                                zaxis: { visible: false, showspikes: false },
                                camera: galaxyCameraRevision >= 0 ? (galaxyCameraRef.current ?? GALAXY_LAUNCH_CAMERA) : GALAXY_LAUNCH_CAMERA,
                                annotations: galaxyArchetypeAnnotations,
                              },
                            }
                          : {
                              xaxis: {
                                title: "PC1",
                                gridcolor: "rgba(0, 212, 224, 0.12)",
                                zerolinecolor: "rgba(0, 212, 224, 0.18)",
                                linecolor: "rgba(0, 212, 224, 0.25)",
                                showline: true,
                                fixedrange: true,
                                ...(plotAxisRange?.x
                                  ? { range: plotAxisRange.x, autorange: false }
                                  : { autorange: true }),
                              },
                              yaxis: {
                                title: "PC2",
                                gridcolor: "rgba(0, 212, 224, 0.12)",
                                zerolinecolor: "rgba(0, 212, 224, 0.18)",
                                linecolor: "rgba(0, 212, 224, 0.25)",
                                showline: true,
                                fixedrange: true,
                                ...(plotAxisRange?.y
                                  ? { range: plotAxisRange.y, autorange: false }
                                  : { autorange: true }),
                              },
                            }),
                        hoverlabel: {
                          bgcolor: "#0A1518",
                          bordercolor: "#00D4E0",
                          font: { family: "JetBrains Mono, monospace", size: galaxyPlotEnabled ? 12 : 11, color: "#DFF3F4" },
                        },
                      }}
                      config={{
                        responsive: true,
                        displayModeBar: false,
                        scrollZoom: false,
                        doubleClick: false,
                      }}
                      style={{ width: "100%", height: "100%" }}
                      onRelayout={(event) => {
                        if (!galaxyPlotEnabled || !event) return;
                        galaxyLastCameraMoveAtRef.current = typeof performance !== "undefined" ? performance.now() : Date.now();
                        galaxyInteractionActiveRef.current = true;
                        window.clearTimeout(galaxyInteractionResumeTimerRef.current);
                        galaxyInteractionResumeTimerRef.current = window.setTimeout(() => {
                          galaxyInteractionActiveRef.current = false;
                        }, GALAXY_CAMERA_INTERACTION_RESUME_MS);
                        const directCamera = event["scene.camera"];
                        if (directCamera?.eye) {
                          const relayoutEyeDistance = clamp(
                            Math.hypot(
                              Number(directCamera.eye?.x ?? 0),
                              Number(directCamera.eye?.y ?? 0),
                              Number(directCamera.eye?.z ?? 0)
                            ),
                            GALAXY_MIN_CAMERA_DISTANCE,
                            GALAXY_MAX_CAMERA_DISTANCE
                          );
                          galaxyCameraRef.current = {
                            ...GALAXY_DEFAULT_CAMERA,
                            ...directCamera,
                            eye: buildStableGalaxyEye(directCamera.eye, relayoutEyeDistance),
                            center: directCamera.center ?? GALAXY_DEFAULT_CAMERA.center,
                            up: directCamera.up ?? GALAXY_DEFAULT_CAMERA.up,
                          };
                          return;
                        }

                        const currentCamera = galaxyCameraRef.current ?? GALAXY_DEFAULT_CAMERA;
                        const hasCameraComponent = Object.keys(event).some((key) => key.startsWith("scene.camera."));
                        if (!hasCameraComponent) return;

                        const nextEye = {
                          x: event["scene.camera.eye.x"] ?? currentCamera.eye?.x ?? GALAXY_DEFAULT_CAMERA.eye.x,
                          y: event["scene.camera.eye.y"] ?? currentCamera.eye?.y ?? GALAXY_DEFAULT_CAMERA.eye.y,
                          z: event["scene.camera.eye.z"] ?? currentCamera.eye?.z ?? GALAXY_DEFAULT_CAMERA.eye.z,
                        };
                        const nextEyeDistance = clamp(
                          Math.hypot(Number(nextEye.x ?? 0), Number(nextEye.y ?? 0), Number(nextEye.z ?? 0)),
                          GALAXY_MIN_CAMERA_DISTANCE,
                          GALAXY_MAX_CAMERA_DISTANCE
                        );
                        galaxyCameraRef.current = {
                          ...GALAXY_DEFAULT_CAMERA,
                          ...currentCamera,
                          eye: buildStableGalaxyEye(nextEye, nextEyeDistance),
                          center: {
                            x: event["scene.camera.center.x"] ?? currentCamera.center?.x ?? GALAXY_DEFAULT_CAMERA.center.x,
                            y: event["scene.camera.center.y"] ?? currentCamera.center?.y ?? GALAXY_DEFAULT_CAMERA.center.y,
                            z: event["scene.camera.center.z"] ?? currentCamera.center?.z ?? GALAXY_DEFAULT_CAMERA.center.z,
                          },
                          up: {
                            x: event["scene.camera.up.x"] ?? currentCamera.up?.x ?? GALAXY_DEFAULT_CAMERA.up.x,
                            y: event["scene.camera.up.y"] ?? currentCamera.up?.y ?? GALAXY_DEFAULT_CAMERA.up.y,
                            z: event["scene.camera.up.z"] ?? currentCamera.up?.z ?? GALAXY_DEFAULT_CAMERA.up.z,
                          },
                        };
                      }}
                      onHover={handleGalaxyPlotHover}
                      onUnhover={handleGalaxyPlotUnhover}
                      onClick={selectGalaxyPointFromPlotClick}
                    />
                  </div>
                </>
              )}
            </div>

            {activeCenterView === "plot" && (
              <div className="cluster-badges">
                {clusterLegendItems.map((item) => (
                  <button
                    key={item.cluster}
                    type="button"
                    className={`cluster-badge ${highlightedCluster === item.cluster ? "highlighted" : ""}`}
                    onClick={() => toggleHighlightedCluster(item.cluster)}
                    style={{
                      "--cluster-color": item.color,
                    }}
                  >
                    <span className="cluster-badge-index">#{item.cluster}</span>
                    <span>{item.count}</span>
                  </button>
                ))}
              </div>
            )}

            {viewTransitionActive && (
              <div className="cluster-view-transition-overlay" aria-hidden="true">
                <div className="cluster-view-transition-core" />
                <div className="cluster-view-transition-scanlines" />
                <div className="cluster-view-transition-static" />
                <div className="cluster-view-transition-flash" />
              </div>
            )}

            {renderGalaxyPlayerProfileOverlay()}
            {renderGalaxyAllFeaturesOverlay()}
          </div>

        </section>

        <div
          className="resize-gutter right-resize-gutter"
          onMouseDown={(event) => startPanelResize("right", event)}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize player report panel"
        >
          <span className="resize-gutter-handle" />
        </div>

        <aside className={`right-panel neon-panel ${rightPanelWidth <= COLLAPSED_PANEL_THRESHOLD && !(galaxyFullscreenEnabled && galaxyPlotEnabled) ? "collapsed-panel" : ""} ${clusterDescriptionViewEnabled ? "cluster-report-panel" : ""} ${careerPathViewEnabled ? "career-report-panel" : ""} ${skillBreakdownViewEnabled || threePtBreakdownViewEnabled ? "skill-breakdown-panel" : ""}`} style={clusterDescriptionViewEnabled ? { "--cluster-color": getClusterColor(highlightedCluster ?? 1) } : undefined}>
          <div className={`report-scroll ${playerReportIdleVisible ? "player-report-idle" : ""}`}>
            <div className="section-header">{clusterDescriptionViewEnabled ? "// CLUSTER_REPORT" : "// PLAYER_REPORT"}</div>

            {clusterDescriptionViewEnabled ? (
              <>
                {highlightedCluster == null && (
                  <div className="empty-state">Click a cluster dot or cluster badge to activate a cluster report.</div>
                )}

                {highlightedCluster != null && (
                  <>
                    <div className="player-headline">
                      <div className="player-name">Cluster {highlightedCluster}</div>
                      <div className="player-meta">
                        {selectedClusterReport?.cluster_size ?? 0} assigned players · {getAlgorithmLabel(currentAlgorithm)} · {getDistanceMetricLabel(selectedDistanceMetric)}
                      </div>
                    </div>

                    <div className="chip-row">
                      <span className="chip">CLUSTER #{highlightedCluster}</span>
                      <span className="chip accent">ROBUST_PROFILE</span>
                    </div>

                    <div className="cluster-report-toggle-row">
                      <button className={`show-all-btn ${clusterSummaryStatMode === "median" ? "active" : ""}`} onClick={() => setClusterSummaryStatMode("median")}>MEDIAN</button>
                      <button className={`show-all-btn ${clusterSummaryStatMode === "average" ? "active" : ""}`} onClick={() => setClusterSummaryStatMode("average")}>AVERAGE</button>
                      <button className={`show-all-btn ${clusterSummaryValueMode === "raw" ? "active" : ""}`} onClick={() => setClusterSummaryValueMode("raw")}>RAW</button>
                      <button className={`show-all-btn ${clusterSummaryValueMode === "percentile" ? "active" : ""}`} onClick={() => setClusterSummaryValueMode("percentile")}>PERCENTILE</button>
                    </div>

                    {loadingClusterReport && <div className="empty-state">Loading cluster report...</div>}
                    {!loadingClusterReport && clusterReportError && <div className="empty-state">{clusterReportError}</div>}

                    {!loadingClusterReport && !clusterReportError && selectedClusterReport && (
                      <>
                        <div className="report-title-row">
                          <div className="report-subtitle report-subtitle-tight">TOP_CLUSTER_FEATURES</div>
                          <button
                            className="show-all-btn"
                            onClick={() => setShowAllFeatures((prev) => !prev)}
                          >
                            {showAllFeatures ? "HIDE_ALL" : "SHOW_ALL"}
                          </button>
                        </div>

                        <div className="feature-summary-list">
                          {selectedClusterReport.top_features.map((item) => (
                            <ClusterFeatureStatCard
                              key={`cluster-top-${item.feature}`}
                              item={item}
                              summaryStatMode={clusterSummaryStatMode}
                              valueMode={clusterSummaryValueMode}
                            />
                          ))}
                        </div>

                        <div className="report-subtitle">LOW_CLUSTER_FEATURES</div>
                        <div className="feature-summary-list">
                          {selectedClusterReport.bottom_features.map((item) => (
                            <ClusterFeatureStatCard
                              key={`cluster-bottom-${item.feature}`}
                              item={item}
                              summaryStatMode={clusterSummaryStatMode}
                              valueMode={clusterSummaryValueMode}
                            />
                          ))}
                        </div>
                      </>
                    )}
                  </>
                )}
              </>
            ) : (
              <>
                {careerPathViewEnabled && selectedCareerMissingSeason && (
                  <>
                    <div className="player-headline">
                      <div className="player-name">{selectedCareerMissingSeason.player_name}</div>
                      <div className="player-meta">
                        {selectedCareerMissingSeason.season} · Did Not Qualify
                      </div>
                    </div>

                    <div className="chip-row">
                      <span className="chip dnq-chip">DID_NOT_QUALIFY</span>
                      <button
                        type="button"
                        className="chip accent career-path-chip"
                        onClick={handleBackToGalaxy}
                      >
                        BACK TO GALAXY
                      </button>
                    </div>

                    <div className="empty-state career-dnq-report">
                      This player-season is not present in the active galaxy view, so no player report is available for this year.
                    </div>
                  </>
                )}

                {!selectedPoint && !selectedCareerMissingSeason && (
                  <div className="empty-state">
                    Click any player point to view player report.
                  </div>
                )}

                {selectedPoint && !selectedCareerMissingSeason && (
                  <>
                    {renderSelectedPlayerProfileHeader()}

                    {loadingDetail && <div className="empty-state">Loading player detail...</div>}

                    {!loadingDetail && selectedDetail && (
                      <>
                        <div className="report-title-row">
                          <div className="report-subtitle report-subtitle-tight">TOP_PERCENTILE_FEATURES</div>
                          <button
                            className="show-all-btn"
                            onClick={() => setShowAllFeatures((prev) => !prev)}
                          >
                            {showAllFeatures ? "HIDE_ALL" : "SHOW_ALL"}
                          </button>
                        </div>

                        <div className="feature-summary-list">
                          {selectedDetail.top_features.map((item) => (
                            <FeatureStatCard
                              key={`top-${item.feature}`}
                              feature={item.feature}
                              value={item.value}
                              percentile={item.percentile}
                              label={item.label}
                            />
                          ))}
                        </div>

                        <div className="report-subtitle">LOW_PERCENTILE_FEATURES</div>
                        <div className="feature-summary-list">
                          {selectedDetail.bottom_features.map((item) => (
                            <FeatureStatCard
                              key={`bottom-${item.feature}`}
                              feature={item.feature}
                              value={item.value}
                              percentile={item.percentile}
                              label={item.label}
                            />
                          ))}
                        </div>
                      </>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        </aside>

        {drawerOpen && (
          <aside className="right-panel neon-panel all-features-panel">
            <div className="report-scroll">
              {clusterDescriptionViewEnabled ? (
                <>
                  <div className="report-title-row drawer-header-row">
                    <div>
                      <div className="section-header">// ALL_CLUSTER_FEATURES</div>
                      <div className="drawer-player-line">Cluster {selectedClusterReport?.cluster_number} · {selectedClusterReport?.cluster_size ?? 0} players</div>
                    </div>
                    <button className="show-all-btn" onClick={() => setShowAllFeatures(false)}>
                      CLOSE
                    </button>
                  </div>

                  <div className="all-features-list">
                    {orderFeaturesByGlossary(selectedClusterReport?.feature_order ?? []).map((feature) => {
                      const item = selectedClusterReport?.feature_summaries?.find((entry) => entry.feature === feature);
                      if (!item) return null;
                      return (
                        <ClusterFeatureStatCard
                          key={`all-cluster-${item.feature}`}
                          item={item}
                          summaryStatMode={clusterSummaryStatMode}
                          valueMode={clusterSummaryValueMode}
                        />
                      );
                    })}
                  </div>
                </>
              ) : (
                <>
                  <div className="report-title-row drawer-header-row">
                    <div>
                      <div className="section-header">// ALL_FEATURES</div>
                      <div className="drawer-player-line">
                        {selectedDetail?.meta?.player_name} · {selectedDetail?.meta?.season} · {selectedDetail?.meta?.teams_played}
                      </div>
                    </div>
                    <button className="show-all-btn" onClick={() => setShowAllFeatures(false)}>
                      CLOSE
                    </button>
                  </div>

                  <div className="all-features-list">
                    {selectedDetail?.stats.map((item) => (
                      <FeatureStatCard
                        key={`all-${item.feature}`}
                        feature={item.feature}
                        value={item.value}
                        percentile={item.percentile}
                        label={item.label}
                      />
                    ))}
                  </div>
                </>
              )}
            </div>
          </aside>
        )}
      </main>
    </div>
  );
}
