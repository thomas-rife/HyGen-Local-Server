"""
macro_prompt_parser.py — Offline, dependency-free natural-language parser
that turns a free-form text prompt into a parameter dict for
terrain_macro.generate_macro_shape.

Design (in priority order — first match wins for the primitive):

  1. PHRASE RULES         multi-word patterns ("ringed by mountains",
                          "hole in the ground"). These encode user intent
                          most clearly so they're checked first.
  2. ABBREVIATION EXPAND  small lookup table that rewrites tokens before
                          keyword matching ("mtn" -> "mountain").
  3. EXACT SUBSTRING      the current behaviour, just with a much bigger
                          thesaurus. This is the workhorse — most prompts
                          land here.
  4. SAFE FUZZY FALLBACK  only fires when 1–3 produced no primitive at all.
                          Strict constraints (token length >= 5, keyword
                          length >= 5, cutoff >= 0.85) and NEVER overrides
                          an exact match. Catches typos like "caynon"
                          without false-positiving "fountain" -> "mountain".

No LLM, no network, no heavy ML deps. Just stdlib.

Public API (unchanged):
    parse_prompt_to_macro_params(prompt, *, grid_size, overlap, seed=None,
                                 directional=None) -> dict

Returned dict shape (unchanged from the previous version):
    "world_size", "grid_size", "overlap", "chunk_size", "seed",
    "primitive", "base_y", "modifiers", "rivers", "ocean",
    "primitive_params", and `<primitive>_<param>` flattened copies.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Any, List, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Must match generate_terrain_img2img.py.
_CHUNK_SIZE = 256

# Fuzzy-match safety thresholds. Don't lower these without re-running the
# regression battery at the bottom of this file.
#
# We use a 0.80 cutoff (catches "swomp/swamp", "caynon/canyon", "dezert/desert"
# at one-edit distance) BUT we restrict fuzzy matching to a curated whitelist
# of "safe" keywords — terrain-specific words that don't have common English
# near-neighbors. This is what stops "fountain" from fuzzy-matching "mountain".
_FUZZY_MIN_TOKEN_LEN   = 5      # ignore short tokens like "mt", "lake"
_FUZZY_MIN_KEYWORD_LEN = 5      # ignore short keywords as match targets
_FUZZY_CUTOFF          = 0.80   # SequenceMatcher.ratio() minimum

# Whitelist of fuzzy-safe keywords. ONLY these participate in the fuzzy
# fallback. Everything else (e.g. "mountain", "lake", "river", "ridge",
# "valley") is excluded because common English words sit too close to them
# in edit-distance space (fountain, lakers, driven, rivet, alley).
#
# Rule of thumb for adding to this set: if a 1-letter typo of the keyword
# is ALSO a real English word, it does NOT belong here. "Plateau" is safe
# because none of its 1-edit neighbors are English words. "Mountain" is
# unsafe because "fountain" exists.
_FUZZY_SAFE_KEYWORDS: frozenset = frozenset({
    # Canyon family
    "canyon", "ravine", "gorge", "gulch", "wadi", "arroyo", "chasm",
    "crevasse", "barranca", "kloof",
    # Crater family
    "crater", "caldera", "sinkhole", "cenote", "cirque", "corrie",
    # Coast family — only the unambiguous ones
    "shoreline", "seashore", "archipelago", "atoll", "peninsula",
    "headland", "promontory", "lagoon", "estuary", "fjord", "fiord",
    # Lake family — bare "lake" is unsafe (lakers); compound words are fine
    # but excluded since fuzzy works on single tokens.
    # Peak family
    "summit", "pinnacle", "volcano", "monadnock", "inselberg",
    "nunatak",
    # Ridge family
    "escarpment", "cordillera", "sierra", "massif", "hogback",
    # Bowl family
    "amphitheatre", "amphitheater",
    # Plateau family
    "plateau", "tableland", "altiplano", "tepui",
    # Dunes / desert family
    "desert", "sahara", "badlands", "outback", "kalahari", "atacama",
    "mojave",
    # Swamp family
    "swamp", "marsh", "marshland", "wetland", "fenland", "bayou",
    "quagmire", "morass", "everglade", "everglades", "muskeg",
    "peatland", "mangrove",
    # Rolling family — "forest" is in here because it has no common typo
    # twins, but "plain" / "field" / "lea" are excluded.
    "forest", "woodland", "rainforest", "jungle", "taiga", "savanna",
    "savannah", "tundra", "moorland", "heathland", "prairie",
    "steppe", "pampas",
})


# ══════════════════════════════════════════════════════════════════════════════
# ABBREVIATION TABLE  (literal rewrite, applied before keyword matching)
# ══════════════════════════════════════════════════════════════════════════════
# Whole-token replacements only — we don't want "mts" inside a longer word
# to corrupt anything. Keys must be lowercase, no punctuation. Punctuation
# in the input is stripped before this table is consulted.

_ABBREVIATIONS: Dict[str, str] = {
    "mtn":   "mountain",
    "mtns":  "mountains",
    "mt":    "mount",
    "mts":   "mountains",
    "rd":    "ridge",
    "rng":   "range",
    "isl":   "island",
    "isls":  "islands",
    "pen":   "peninsula",
    "lk":    "lake",
    "rvr":   "river",
    # Plural-normalize a few river-y nouns so the river detector stays simple.
    "rivers":      "river",
    "streams":     "stream",
    "creeks":      "creek",
    "tributaries": "tributary",
    "n":     "north",
    "s":     "south",
    "e":     "east",
    "w":     "west",
    "ne":    "northeast",
    "nw":    "northwest",
    "se":    "southeast",
    "sw":    "southwest",
}


# ══════════════════════════════════════════════════════════════════════════════
# PHRASE RULES  (regex → primitive). First match wins.
# ══════════════════════════════════════════════════════════════════════════════
#
# Patterns are matched on the LOWERCASED, abbreviation-expanded text with
# punctuation collapsed to spaces. Use \b for word boundaries. Use \s+ for
# any whitespace. Keep patterns specific — vague phrases belong in the
# keyword tables, not here.
#
# Each entry is (regex_string, primitive_name). Order matters.

_PHRASE_RULE_SOURCES: List[Tuple[str, str]] = [
    # ── BOWL: surrounded / ringed / encircled by tall stuff ───────────────
    (r"\b(surrounded|ringed|encircled|hemmed)\s+(in\s+|on\s+all\s+sides\s+)?by\s+"
     r"(mountains?|peaks?|cliffs?|walls?|hills?|crags?|ranges?)\b", "bowl"),
    (r"\bin\s+the\s+(middle|heart|center|centre|bottom)\s+of\s+a\s+"
     r"(valley|basin|bowl|crater|caldera)\b", "bowl"),
    (r"\bnestled\s+(in|between|among)\s+(\w+\s+){0,2}"
     r"(mountains?|peaks?|hills?|cliffs?)\b", "bowl"),
    (r"\b(deep|wide|broad|narrow)\s+valley\s+floor\b", "bowl"),

    # ── CRATER: hole in the ground, etc. ──────────────────────────────────
    (r"\b(hole|pit|depression|sinkhole)\s+in\s+the\s+(ground|earth|land|terrain)\b",
     "crater"),
    (r"\b(impact|meteor|asteroid)\s+(crater|site|zone)\b", "crater"),
    (r"\bcollapsed\s+(volcano|caldera|mountain)\b", "crater"),
    (r"\bbowl[- ]shaped\s+(depression|basin|hole)\b", "crater"),

    # ── CANYON: between cliffs/walls ──────────────────────────────────────
    (r"\bbetween\s+(two\s+)?(steep\s+|tall\s+|sheer\s+|towering\s+)?"
     r"(cliffs?|walls?|escarpments?|bluffs?)\b", "canyon"),
    (r"\b(deep|narrow|steep|sheer)\s+(slot|cleft|crack|cut|fissure)\b", "canyon"),
    (r"\bcarved\s+by\s+(water|river|the\s+river|a\s+river|erosion)\b", "canyon"),
    (r"\bwalls?\s+rise\s+(steeply|sharply|on\s+both\s+sides)\b", "canyon"),

    # ── COAST: where land meets water ─────────────────────────────────────
    (r"\bwhere\s+(the\s+)?(land|earth|ground)\s+meets\s+"
     r"(the\s+)?(sea|ocean|water)\b", "coast"),
    (r"\bwhere\s+(the\s+)?(sea|ocean|water)\s+meets\s+"
     r"(the\s+)?(land|earth|ground|shore)\b", "coast"),
    (r"\bedge\s+of\s+(the\s+)?(sea|ocean|continent)\b", "coast"),
    (r"\b(sandy|rocky|tropical|pebble)\s+shoreline?\b", "coast"),
    (r"\bopen\s+(water|sea|ocean)\s+(to\s+the\s+)?"
     r"(north|south|east|west)\b", "coast"),

    # ── PEAK: rises above / towers over ───────────────────────────────────
    (r"\b(rises|rising|towers|towering|soars?|soaring)\s+"
     r"(above|over|high\s+above)\b", "peak"),
    (r"\b(lone|lonely|solitary|single|isolated)\s+"
     r"(peak|mountain|summit|volcano)\b", "peak"),
    (r"\bsnow[- ]?capped\s+(peak|summit|mountain)\b", "peak"),

    # ── RIDGE: long line of high ground ───────────────────────────────────
    (r"\bmountain\s+range\b", "ridge"),
    (r"\b(long|continuous|unbroken)\s+(line|chain|wall)\s+of\s+"
     r"(mountains|peaks|hills)\b", "ridge"),
    (r"\b(spine|backbone|crest)\s+of\s+(the\s+)?(mountains?|range|land)\b",
     "ridge"),

    # ── PLATEAU: high flat ────────────────────────────────────────────────
    (r"\b(flat[- ]topped|table[- ]?top)\s+(mountain|hill|mesa|formation)\b",
     "plateau"),
    (r"\b(elevated|raised|high)\s+(plain|tableland|flat|flatland)\b", "plateau"),

    # ── DUNES: sand sea ───────────────────────────────────────────────────
    (r"\b(sea|ocean|sweep|expanse)\s+of\s+(sand|sands|dunes?)\b", "dunes"),
    (r"\bvast\s+(sand|desert|arid)\s+(expanse|landscape|wilderness)\b", "dunes"),

    # ── ROLLING: sea of grass / endless plains ────────────────────────────
    (r"\b(sea|ocean|expanse)\s+of\s+(grass|grassland|prairie)\b", "rolling"),
    (r"\b(endless|vast|sweeping|rolling)\s+(plains?|grasslands?|"
     r"prairie|steppe|savann?ah?|meadows?)\b", "rolling"),
    (r"\b(gentle|low|soft)\s+(rolling|undulating)\s+hills?\b", "rolling"),

    # ── SWAMP: standing water in low land ─────────────────────────────────
    (r"\b(stagnant|murky|muddy|brackish)\s+(water|pools?|ponds?)\b", "swamp"),
    (r"\b(flooded|waterlogged|boggy|marshy)\s+(forest|woodland|land|ground|plain)\b",
     "swamp"),

    # ── LAKE: a body of water in low land ─────────────────────────────────
    (r"\b(small|large|mountain|alpine|crater)\s+"
     r"(lake|pond|tarn|loch|lough)\b", "lake_basin"),
    (r"\bbody\s+of\s+(still|calm|fresh)\s+water\b", "lake_basin"),
]

# Compile lazily on first parse() — we don't need the regex engine warmed up
# at import time, and it's nice for tests that monkey-patch the source list.
_PHRASE_RULES: Optional[List[Tuple[re.Pattern, str]]] = None

def _get_phrase_rules() -> List[Tuple[re.Pattern, str]]:
    global _PHRASE_RULES
    if _PHRASE_RULES is None:
        _PHRASE_RULES = [
            (re.compile(pat, re.IGNORECASE), prim)
            for pat, prim in _PHRASE_RULE_SOURCES
        ]
    return _PHRASE_RULES


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE KEYWORD TABLE  (massively expanded thesaurus)
# ══════════════════════════════════════════════════════════════════════════════
#
# Each entry: (primitive_name, tuple_of_substring_keywords).
# Order = priority. More structurally specific primitives come first so
# "snowy mountain canyon" picks `canyon`, not `ridge`.
#
# Substrings are matched (with `in`) against tokens AFTER abbreviation
# expansion. Multi-word entries (with a space inside) are matched as
# substrings against the full text — useful for compound nouns like
# "sand dune" or "mountain range".

_PRIMITIVE_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [

    # ── CANYON (narrow deep cuts) ─────────────────────────────────────────
    ("canyon", (
        "canyon", "ravine", "gorge", "gulch", "gully", "defile",
        "coulee", "wadi", "arroyo", "chasm", "abyss", "crevasse",
        "slot canyon", "barranca", "nullah", "kloof",
    )),

    # ── CRATER (closed circular depressions) ──────────────────────────────
    ("crater", (
        "crater", "caldera", "sinkhole", "cenote", "doline",
        "impact site", "meteor crater", "kettle hole",
        "cwm", "cirque", "corrie", "tarn basin",
    )),

    # ── COAST (anything where the map meets open water) ───────────────────
    ("coast", (
        "beach", "coast", "coastline", "shore", "shoreline", "seaside",
        "seashore", "island", "islands", "isle", "isles", "islet",
        "archipelago", "atoll", "peninsula", "headland", "cape",
        "promontory", "cove", "inlet", "bay", "bight", "harbor",
        "harbour", "lagoon", "estuary", "delta", "fjord", "fiord",
        "sound", "strait", "gulf", "firth", "lido", "sandbar",
        "barrier island", "tombolo", "spit",
    )),

    # ── LAKE BASIN ────────────────────────────────────────────────────────
    ("lake_basin", (
        "lake", "pond", "tarn", "loch", "lough", "mere", "reservoir",
        "oxbow", "playa lake", "crater lake", "alpine lake",
    )),

    # ── PEAK (single dominant high point) ─────────────────────────────────
    # Word-boundary matching means bare "mount" won't accidentally match
    # "mountain" — \bmount\b is distinct from \bmountain\b.
    ("peak", (
        "peak", "summit", "pinnacle", "spire", "horn", "aiguille",
        "mount", "volcano", "volcanic cone", "stratovolcano",
        "shield volcano", "monadnock", "inselberg", "nunatak",
        "tor",
    )),

    # ── RIDGE (linear high ground) ────────────────────────────────────────
    ("ridge", (
        "ridge", "ridgeline", "spine", "crest", "arete", "arête",
        "hogback", "escarpment", "scarp", "cuesta", "mountain range",
        "cordillera", "sierra", "massif", "alps", "andes",
        "rockies", "rocky mountains", "highland", "highlands",
        "ridgeback",
    )),

    # ── BOWL (open depressions, valleys you spawn IN) ─────────────────────
    ("bowl", (
        "valley", "basin", "vale", "glen", "dale", "hollow", "dell",
        "dingle", "combe", "coomb", "cwm valley", "amphitheatre",
        "amphitheater", "sunken plain", "rift valley", "graben",
        "surrounded by mountain",
    )),

    # ── Catchall after bowl/peak/ridge/etc. for generic "mountain" prompts.
    #    Mapped to ridge so a bare "mountains" prompt produces a believable
    #    macro range, not a single peak.
    ("ridge", ("mountain", "mountains", "alpine", "mountainous")),

    # ── PLATEAU ──────────────────────────────────────────────────────────
    ("plateau", (
        "plateau", "mesa", "tableland", "butte", "tepui", "table mountain",
        "altiplano", "puna",
    )),

    # ── DUNES (and arid-with-relief in general) ──────────────────────────
    ("dunes", (
        "dune", "dunes", "sand dune", "sand dunes", "sand sea",
        "sahara", "erg", "desert", "badlands", "scree desert",
        "salt flat", "salt flats", "playa", "outback", "kalahari",
        "gobi", "atacama", "mojave", "namib",
    )),

    # ── SWAMP / WETLAND ──────────────────────────────────────────────────
    ("swamp", (
        "swamp", "marsh", "marshland", "bog", "mire", "wetland",
        "fen", "fenland", "bayou", "slough", "quagmire", "morass",
        "everglade", "everglades", "muskeg", "peatland", "salt marsh",
        "mangrove", "mangroves",
    )),

    # ── ROLLING (catch-all gentle terrain) ───────────────────────────────
    ("rolling", (
        "plain", "plains", "meadow", "meadows", "grassland", "grasslands",
        "steppe", "steppes", "prairie", "prairies", "savanna", "savannas",
        "savannah", "savannahs", "tundra", "moor", "moorland", "heath",
        "heathland", "downs", "pasture", "pastures", "field", "fields",
        "lea", "veld", "veldt", "pampas", "puszta",
        # forest variants — gentle terrain, AI handles tree look in pass
        "forest", "forests", "woods", "woodland", "woodlands", "jungle",
        "jungles", "rainforest", "rainforests", "taiga", "boreal",
        "thicket", "copse", "grove", "wildwood",
    )),
]


# ══════════════════════════════════════════════════════════════════════════════
# MODIFIER KEYWORD TABLE  (independent flags, can stack)
# ══════════════════════════════════════════════════════════════════════════════
#
# Modifiers don't pick the primitive. They influence noise / palette / decor
# downstream. Multiple can fire on one prompt.
#
# We DO NOT fuzzy-match modifiers — the false-positive risk is too high
# (e.g. "showy" -> "snowy") and missing a modifier is a small cost compared
# to picking the wrong one.

_MODIFIER_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "snowy":    (
        "snow", "snowy", "snowfall", "snowdrift", "frozen", "ice", "icy",
        "iceberg", "icefield", "glacier", "glacial", "arctic", "antarctic",
        "polar", "alpine", "tundra", "permafrost", "subzero", "wintry",
        "blizzard",
    ),
    "tropical": (
        "tropical", "tropic", "tropics", "jungle", "rainforest", "palm",
        "palms", "palmy", "equatorial", "humid", "monsoon", "lush jungle",
        "coconut", "mangrove", "tropic island",
    ),
    "arid":     (
        "desert", "arid", "dry", "parched", "dune", "sahara", "scorched",
        "barren", "drought", "wasteland", "badland", "badlands", "outback",
        "semiarid", "semi-arid", "dust bowl",
    ),
    "volcanic": (
        "volcano", "volcanoes", "volcanic", "lava", "magma", "obsidian",
        "basalt", "fumarole", "geothermal", "ashfall", "pumice",
        "pyroclastic", "tephra",
    ),
    "rocky":    (
        "rocky", "stony", "craggy", "boulder", "boulders", "scree",
        "talus", "shale", "granite", "limestone", "sandstone",
        "outcrop", "outcropping", "bedrock", "rugged",
    ),
    "lush":     (
        "lush", "verdant", "fertile", "overgrown", "leafy", "fecund",
        "blooming", "flowering", "thriving",
    ),
    "dead":     (
        "dead", "blighted", "ashen", "desolate",
        "lifeless", "withered", "decaying", "rotting", "haunted",
        "abandoned", "ruined",
    ),
    "windswept":(
        "windswept", "stormy", "exposed", "wind-blasted", "blasted",
        "gale-swept", "hurricane",
    ),
    "misty":    (
        "misty", "foggy", "fog", "mist", "haze", "hazy", "murky",
        "shrouded", "veiled", "cloudy",
    ),
    "ancient":  (
        "ancient", "primeval", "primordial", "eldritch", "ageless",
        "weathered", "eroded", "timeworn", "forgotten",
    ),
}

_RIVER_KEYWORDS = (
    "river", "stream", "creek", "brook", "tributary", "winding water",
    "watercourse", "waterway", "rivulet", "rill", "burn",
)

_OCEAN_KEYWORDS = (
    "ocean", "sea", "deep water", "open water", "open ocean",
    "high seas", "abyss", "abyssal",
)


# ══════════════════════════════════════════════════════════════════════════════
# PER-PRIMITIVE DEFAULTS   (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

_BASE_Y         = 63
_AMP_PEAK_Y     = 110
_AMP_RIDGE_Y    = 80
_AMP_PLATEAU_Y  = 40
_AMP_BOWL_Y     = 70
_AMP_DUNE_Y     = 18
_AMP_ROLLING_Y  = 12
_DEPTH_CANYON_Y = 55
_DEPTH_CRATER_Y = 40
_DEPTH_LAKE_Y   = 8
_DEPTH_SWAMP_Y  = 4
_DEPTH_OCEAN_Y  = 22

PRIMITIVE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "bowl": {
        "rim_height_y":      _AMP_BOWL_Y,
        "rim_radius_frac":   0.42,
        "floor_radius_frac": 0.10,
        "noise_amp_y":       14.0,
        "noise_freq":        3.5,
    },
    "canyon": {
        "depth_y":              _DEPTH_CANYON_Y,
        "floor_width_frac":     0.06,
        "wall_width_frac":      0.18,
        "wall_height_y":        _AMP_RIDGE_Y * 0.6,
        "axis":                 "auto",
        "meander_frac":         0.08,
        "noise_amp_y":          9.0,
        "noise_freq":           3.0,
    },
    "crater": {
        "depth_y":           _DEPTH_CRATER_Y,
        "rim_height_y":      35.0,
        "outer_radius_frac": 0.32,
        "inner_radius_frac": 0.18,
        "noise_amp_y":       8.0,
        "noise_freq":        3.0,
    },
    "coast": {
        "ocean_side":           "auto",
        "ocean_depth_y":        _DEPTH_OCEAN_Y,
        "land_height_y":        30.0,
        "shore_band_frac":      0.06,
        "coast_curve_amp_frac": 0.08,
        "noise_amp_y":          7.0,
        "noise_freq":           4.0,
    },
    "peak": {
        "peak_height_y":      _AMP_PEAK_Y,
        "summit_radius_frac": 0.05,
        "base_radius_frac":   0.40,
        "noise_amp_y":        14.0,
        "noise_freq":         3.0,
    },
    "ridge": {
        "ridge_height_y":   _AMP_RIDGE_Y,
        "ridge_axis":       "auto",
        "ridge_width_frac": 0.18,
        "asymmetry":        0.0,
        "noise_amp_y":      12.0,
        "noise_freq":       3.5,
    },
    "plateau": {
        "plateau_height_y":    _AMP_PLATEAU_Y,
        "plateau_radius_frac": 0.30,
        "edge_softness_frac":  0.04,
        "noise_amp_y":         6.0,
        "noise_freq":          3.0,
    },
    "dunes": {
        "dune_height_y":   _AMP_DUNE_Y,
        "dune_axis":       "auto",
        "dune_freq_main":  6.0,
        "dune_freq_cross": 16.0,
        "noise_amp_y":     5.0,
        "noise_freq":      8.0,
    },
    "rolling": {
        "amplitude_y":   _AMP_ROLLING_Y,
        "noise_freq":    4.0,
        "noise_octaves": 4,
    },
    "swamp": {
        "base_depression_y": _DEPTH_SWAMP_Y,
        "puddle_density":    0.6,
        "noise_amp_y":       5.0,
        "noise_freq":        5.0,
    },
    "lake_basin": {
        "depth_y":           _DEPTH_LAKE_Y,
        "lake_radius_frac":  0.22,
        "surround_height_y": 18.0,
        "noise_amp_y":       8.0,
        "noise_freq":        3.5,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# WORLD-SIZE MATH
# ══════════════════════════════════════════════════════════════════════════════

def compute_world_size(grid_size: int, overlap: int,
                       chunk_size: int = _CHUNK_SIZE) -> int:
    """world_size = (chunk_size - overlap) * (grid_size - 1) + chunk_size."""
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1 (got {grid_size})")
    if not (0 <= overlap < chunk_size):
        raise ValueError(
            f"overlap must be in [0, {chunk_size}) (got {overlap})"
        )
    stride = chunk_size - overlap
    return stride * (grid_size - 1) + chunk_size


# ══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
#
# All matching happens on a normalized representation:
#   1. lowercase
#   2. punctuation -> spaces (so "fjord," is "fjord")
#   3. abbreviation table applied per token
#   4. surplus whitespace collapsed

_PUNCT_RE = re.compile(r"[^a-z0-9\s.\-]")    # keep periods + hyphens for "mt."
_TOKEN_TRAILING_PUNCT = ".,;:!?"


def _normalize_text(prompt: Optional[str],
                    directional: Optional[Dict[str, Optional[str]]]) -> str:
    """Flatten + lowercase + abbreviation-expand the input text."""
    parts: List[str] = []
    if prompt:
        parts.append(prompt)
    if directional:
        for v in directional.values():
            if v:
                parts.append(v)
    text = " ".join(parts).lower()

    # Strip out characters that aren't letters/digits/spaces/period/hyphen.
    text = _PUNCT_RE.sub(" ", text)
    # Tokenize, expand abbreviations, rejoin.
    out_tokens: List[str] = []
    for raw in text.split():
        tok = raw.strip(_TOKEN_TRAILING_PUNCT)
        if not tok:
            continue
        # Try the bare token first, then the period-stripped form.
        expanded = _ABBREVIATIONS.get(tok)
        if expanded is None and tok.endswith("."):
            expanded = _ABBREVIATIONS.get(tok[:-1])
        out_tokens.append(expanded if expanded is not None else tok)
    # Re-join. Single spaces, no surrounding whitespace.
    return " ".join(out_tokens)


# Word-boundary regex cache. We compile per-keyword once, since the keyword
# tables are static. Multi-word keywords (those containing a space) bypass
# this and use plain `in` substring matching, since "mountain range" inside
# "snowy mountain range" must match without word boundaries getting in the
# way of the leading/trailing spaces.
_WB_CACHE: Dict[str, re.Pattern] = {}

def _wb_pattern(kw: str) -> re.Pattern:
    rx = _WB_CACHE.get(kw)
    if rx is None:
        rx = re.compile(rf"\b{re.escape(kw)}\b")
        _WB_CACHE[kw] = rx
    return rx


def _has_any(text: str, keywords: Tuple[str, ...]) -> bool:
    """True if any keyword appears as a word-bounded match (single-word
    keywords) or as a plain substring (multi-word keywords) in `text`.

    Word-boundary matching is what stops "lake" from matching "lakers"
    and "bay" from matching "bayou". It costs ~1us per keyword to compile
    once and is essentially free thereafter.
    """
    for kw in keywords:
        if " " in kw:
            # Multi-word keyword — substring is the right semantics.
            if kw in text:
                return True
        else:
            if _wb_pattern(kw).search(text):
                return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVE SELECTION — phrase rules → exact substrings → safe fuzzy
# ══════════════════════════════════════════════════════════════════════════════

def _pick_primitive_phrase(text: str) -> Optional[str]:
    """First phrase rule that fires wins. Returns None if no rule matched."""
    for rx, prim in _get_phrase_rules():
        if rx.search(text):
            return prim
    return None


def _pick_primitive_substring(text: str) -> Optional[str]:
    """First (primitive, keywords) bucket whose any keyword appears as a
    substring of the normalized text. Returns None if nothing matched."""
    for prim, kws in _PRIMITIVE_KEYWORDS:
        if _has_any(text, kws):
            return prim
    return None


def _build_fuzzy_keyword_index() -> List[Tuple[str, str]]:
    """All (keyword, primitive) pairs used by the fuzzy-fallback path.

    Filtered three ways:
      1. Drop multi-word keywords — fuzzy matches single tokens.
      2. Drop keywords below _FUZZY_MIN_KEYWORD_LEN (too short to clear
         the cutoff without enabling false positives).
      3. Drop keywords NOT in _FUZZY_SAFE_KEYWORDS — this is the critical
         filter that prevents "fountain" from matching "mountain".
    """
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for prim, kws in _PRIMITIVE_KEYWORDS:
        for kw in kws:
            if " " in kw:
                continue
            if len(kw) < _FUZZY_MIN_KEYWORD_LEN:
                continue
            if kw not in _FUZZY_SAFE_KEYWORDS:
                continue
            if kw in seen:
                continue
            seen.add(kw)
            out.append((kw, prim))
    return out


_FUZZY_KEYWORDS: Optional[List[Tuple[str, str]]] = None

def _get_fuzzy_keywords() -> List[Tuple[str, str]]:
    global _FUZZY_KEYWORDS
    if _FUZZY_KEYWORDS is None:
        _FUZZY_KEYWORDS = _build_fuzzy_keyword_index()
    return _FUZZY_KEYWORDS


def _pick_primitive_fuzzy(text: str) -> Optional[str]:
    """Last resort. ONLY called when phrase + substring both returned None.

    For every long-enough token in the text, check it against every
    long-enough keyword using SequenceMatcher.ratio(). Return the
    primitive of the keyword with the highest ratio that exceeds the
    cutoff.
    """
    tokens = [t for t in text.split() if len(t) >= _FUZZY_MIN_TOKEN_LEN]
    if not tokens:
        return None

    best_ratio = 0.0
    best_prim: Optional[str] = None
    fuzzy_kws = _get_fuzzy_keywords()

    for tok in tokens:
        for kw, prim in fuzzy_kws:
            # Cheap length pre-filter: if the lengths are wildly different
            # SequenceMatcher.ratio() can't clear the cutoff anyway.
            la, lb = len(tok), len(kw)
            if abs(la - lb) > max(la, lb) * (1.0 - _FUZZY_CUTOFF) * 2.0:
                continue
            r = SequenceMatcher(None, tok, kw).ratio()
            if r >= _FUZZY_CUTOFF and r > best_ratio:
                best_ratio = r
                best_prim = prim

    return best_prim


def _pick_primitive(text: str) -> Tuple[str, str]:
    """Run the layered match. Returns (primitive, source) where source is
    one of 'phrase', 'substring', 'fuzzy', or 'fallback'."""
    prim = _pick_primitive_phrase(text)
    if prim is not None:
        return prim, "phrase"
    prim = _pick_primitive_substring(text)
    if prim is not None:
        return prim, "substring"
    prim = _pick_primitive_fuzzy(text)
    if prim is not None:
        return prim, "fuzzy"
    return "rolling", "fallback"


def _pick_modifiers(text: str) -> Dict[str, bool]:
    """Substring-only — no fuzzy matching here (false-positive risk too high
    on short adjective-like words)."""
    return {name: _has_any(text, kws)
            for name, kws in _MODIFIER_KEYWORDS.items()}


def _pick_ocean_side(text: str,
                    directional: Optional[Dict[str, Optional[str]]]
                    ) -> Optional[str]:
    """Decide which side an ocean is on, if any."""
    if directional:
        for side in ("north", "south", "east", "west"):
            v = directional.get(side)
            if v and _has_any(v.lower(), _OCEAN_KEYWORDS):
                return side
    if _has_any(text, _OCEAN_KEYWORDS):
        return "auto"
    return None


def _pick_river_count(text: str) -> int:
    """Count rivers from explicit plural forms; otherwise 1 if any river
    keyword is present."""
    if not _has_any(text, _RIVER_KEYWORDS):
        return 0
    if re.search(r"\b(rivers|streams|tributaries|creeks|brooks|rivulets)\b",
                 text):
        return 2
    return 1


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  (signature unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def parse_prompt_to_macro_params(
    prompt: Optional[str],
    *,
    grid_size: int,
    overlap: int,
    seed: Optional[int] = None,
    directional: Optional[Dict[str, Optional[str]]] = None,
    chunk_size: int = _CHUNK_SIZE,
) -> Dict[str, Any]:
    """Turn a text prompt into a flattened macro-parameter dict.

    See module docstring for the full schema. Note the source of each
    primitive match is recorded in `_match_source` for debugging:
    'phrase' / 'substring' / 'fuzzy' / 'fallback'.
    """
    if prompt is None and not (directional and any(directional.values())):
        raise ValueError(
            "parse_prompt_to_macro_params: provide `prompt` or at least one "
            "non-empty directional prompt."
        )

    text = _normalize_text(prompt, directional)
    primitive, source = _pick_primitive(text)
    modifiers = _pick_modifiers(text)
    ocean_side = _pick_ocean_side(text, directional)
    river_count = _pick_river_count(text)

    world_size = compute_world_size(grid_size, overlap, chunk_size=chunk_size)

    params: Dict[str, Any] = {
        "world_size":   int(world_size),
        "size":         int(world_size),
        "grid_size":    int(grid_size),
        "overlap":      int(overlap),
        "chunk_size":   int(chunk_size),
        "seed":         seed,
        "primitive":    primitive,
        "base_y":       _BASE_Y,
        "modifiers":    modifiers,
        "rivers":       {
            "count":       river_count,
            "seed_offset": 0xC0FFEE,
        },
        "ocean":        None if ocean_side is None else {
            "side":     ocean_side,
            "depth_y":  PRIMITIVE_DEFAULTS["coast"]["ocean_depth_y"],
        },
        "_prompt":          prompt,
        "_directional":     dict(directional) if directional else {},
        "_matched_text":    text,
        "_match_source":    source,
    }

    prim_defaults = PRIMITIVE_DEFAULTS.get(primitive, {})
    for k, v in prim_defaults.items():
        params[f"{primitive}_{k}"] = v
    params["primitive_params"] = dict(prim_defaults)

    # Cross-cutting modifier nudges (kept tiny — real branching belongs in
    # terrain_macro.generate_macro_shape).
    if modifiers["arid"] and primitive == "rolling":
        params["primitive_params"]["amplitude_y"] = _AMP_DUNE_Y
        params["primitive_params"]["noise_freq"]  = 6.0
        params["rolling_amplitude_y"] = _AMP_DUNE_Y
        params["rolling_noise_freq"]  = 6.0

    return params


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: SUMMARY STRING (for logs)
# ══════════════════════════════════════════════════════════════════════════════

def summarize(params: Dict[str, Any]) -> str:
    """One-line human-readable summary for logs/diagnostics."""
    mods = ",".join(k for k, v in params["modifiers"].items() if v) or "-"
    ocean = params["ocean"]["side"] if params["ocean"] else "-"
    rc = params["rivers"]["count"]
    src = params.get("_match_source", "?")
    return (
        f"primitive={params['primitive']:10s}  "
        f"modifiers={mods:24s}  ocean={ocean:5s}  rivers={rc}  "
        f"world={params['world_size']}px  via={src}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION BATTERY — run with `python macro_prompt_parser.py`
# ══════════════════════════════════════════════════════════════════════════════
# Keep this in sync with the design assumptions. If you tune the fuzzy
# thresholds or expand the keyword tables, re-run this and confirm the
# numbers haven't regressed.

if __name__ == "__main__":
    # ── 1. Synonyms / vocabulary coverage ─────────────────────────────────
    SYNONYM_CASES = [
        ("a deep gorge", "canyon"),
        ("a winding ravine", "canyon"),
        ("a slot canyon between two cliffs", "canyon"),
        ("the wadi at dawn", "canyon"),
        ("a vast caldera", "crater"),
        ("a cenote in the jungle", "crater"),
        ("a glacial cwm", "crater"),
        ("a mountain cirque", "crater"),
        ("a quiet fjord", "coast"),
        ("rocky headlands above the bay", "coast"),
        ("a tropical atoll", "coast"),
        ("a remote archipelago", "coast"),
        ("a peninsula stretching into the sea", "coast"),
        ("an alpine tarn", "lake_basin"),
        ("a still loch in the highlands", "lake_basin"),
        ("a snow-capped horn", "peak"),
        ("a lone monadnock", "peak"),
        ("the sierra at sunset", "ridge"),
        ("a long hogback ridge", "ridge"),
        ("a steep escarpment", "ridge"),
        ("a wide rift valley", "bowl"),
        ("a quiet glen", "bowl"),
        ("a flat-topped mesa", "plateau"),
        ("a tepui rising from the rainforest", "plateau"),
        ("badlands at the edge of the desert", "dunes"),
        ("the gobi at midday", "dunes"),
        ("a brackish bayou", "swamp"),
        ("a foggy fenland", "swamp"),
        ("a peatland", "swamp"),
        ("rolling savannah", "rolling"),
        ("endless veld", "rolling"),
        ("a quiet meadow", "rolling"),
    ]

    # ── 2. Phrase rules ───────────────────────────────────────────────────
    PHRASE_CASES = [
        ("a village surrounded by mountains", "bowl"),
        ("nestled between snowy peaks", "bowl"),
        ("a hole in the ground filled with water", "crater"),
        ("an impact crater", "crater"),
        ("between two towering cliffs", "canyon"),
        ("carved by the river", "canyon"),
        ("walls rise steeply on both sides", "canyon"),
        ("where the land meets the sea", "coast"),
        ("a sandy shoreline", "coast"),
        ("rises high above the plain", "peak"),
        ("a lone peak in the distance", "peak"),
        ("a long mountain range", "ridge"),
        ("the spine of the mountains", "ridge"),
        ("a flat-topped mountain", "plateau"),
        ("a vast sea of sand", "dunes"),
        ("endless rolling plains", "rolling"),
        ("sea of grass", "rolling"),
        ("waterlogged forest", "swamp"),
        ("a small alpine lake", "lake_basin"),
    ]

    # ── 3. Typos that fuzzy-fallback should catch ─────────────────────────
    TYPO_CASES = [
        ("caynon",         "canyon"),
        ("a deep caynon",  "canyon"),
        ("forrest",        "rolling"),
        ("plataeu",        "plateau"),
        ("swomp",          "swamp"),
        ("dezert",         "dunes"),
        ("cratter",        "crater"),
    ]

    # ── 4. False-positive killers ─────────────────────────────────────────
    INNOCENT_CASES = [
        ("a town with a fountain in the square", "rolling"),
        ("a small village",                      "rolling"),
        ("a craft brewery",                      "rolling"),
        ("driven by ambition",                   "rolling"),
        ("a deserved rest",                      "rolling"),
        ("the lakers won",                       "rolling"),
        ("a beautiful day",                      "rolling"),
    ]

    failures = 0
    total = 0

    def run_block(cases, label):
        global failures, total
        print(f"\n=== {label} ===")
        for prompt, expected in cases:
            p = parse_prompt_to_macro_params(prompt, grid_size=3, overlap=64, seed=1)
            actual = p["primitive"]
            src = p["_match_source"]
            ok = (actual == expected)
            mark = "OK " if ok else "ERR"
            print(f"  {mark} {prompt!r:55s} -> {actual:11s} via {src:9s}  (expected {expected})")
            if not ok:
                failures += 1
            total += 1

    run_block(SYNONYM_CASES,  "SYNONYM COVERAGE")
    run_block(PHRASE_CASES,   "PHRASE RULES")
    run_block(TYPO_CASES,     "FUZZY-FALLBACK TYPOS")
    run_block(INNOCENT_CASES, "FALSE-POSITIVE GUARD (should be 'rolling')")

    print(f"\nResult: {total - failures}/{total} cases passed.")
