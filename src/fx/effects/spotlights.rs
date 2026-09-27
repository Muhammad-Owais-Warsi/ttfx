//! spotlights on the fx engine (old engine: effects/spotlights.rs).
//!
//! The spotlights are added, never-shown characters; each walks its eleven
//! chained bezier paths ("0".."10", looping) until the search ends, then its
//! "center" path. Each frame the input characters inside any spotlight's
//! ellipse show their bright colors, dimmed towards the beam's edge; the rest
//! fall back to their dark colors. A frame draws nothing from the RNG and
//! only sets appearances, so the lit set is a list plus a per-character
//! frame stamp and its order does not matter.
//!
//! Differences from the old engine:
//! - each character keeps its bright, dark and override visuals;
//! - characters surely inside the edge by exact integer distance skip the
//!   hypot fold; beam distances come from a hypot memo over (|dx|, |dy|);
//! - dimmed visuals are memoized per character (last factor) and by (bright
//!   visual, factor bits), which fix the symbol and both adjusted colors;
//! - a frame whose spotlights did not move is skipped.

use std::collections::HashMap;

use crate::effects::spotlights::SpotlightsConfig;
use crate::engine::animation::{Animation, ExistingColorHandling};
use crate::engine::error::EngineError;
use crate::engine::terminal::{CharacterFilter, CharacterSort};
use crate::fx::run::Effect;
use crate::fx::visual::{VisualInfo, BOLD, HAS_COLORS};
use crate::fx::{At, Engine, FxBuild, Hooks, Name, Sym, Visual, NONE};
use crate::utils::easing::Easing;
use crate::utils::geometry::{self, Coord};
use crate::utils::graphics::{Color, Gradient};
use crate::utils::pycompat::{floor_div, round_half_even};

/// _is_spotlightable.
const LIT: u8 = 1;
/// Shows its input colors whatever it is given (--existing-color-handling
/// always): every appearance is the bright visual.
const FIXED: u8 = 2;

/// Per spotlightable character, in column order (see `order`).
#[derive(Clone, Copy)]
struct Rec {
    sym: Sym,
    bright: Visual,
    dark: Visual,
    /// _get_expand_color_override's visual, NONE when it does not apply.
    over: Visual,
    /// The bright pair's index in `pairs`.
    pair: u32,
    /// The bright visual's dense index for `dense` (NONE: not there).
    bidx: u32,
    /// The symbol's dense index for `vdense` (NONE: not there).
    sidx: u32,
    /// The last brightness factor's visual (NONE: none yet) and bits.
    dimmed: Visual,
    factor: u64,
    flags: u8,
}

/// The dense search-phase memo's limits: factors, and bright visuals.
const DENSE_FACTORS: usize = 256;
const DENSE_BRIGHTS: usize = 4096;
const FMAP_BITS: u32 = 9;
/// The adjusted pairs' dense table: at most this many pairs, and symbols.
const ADJUSTED_BITS: u32 = 16;
const ADJUSTED_PAIRS: usize = 1 << (ADJUSTED_BITS - 1);
const DENSE_SYMS: usize = 64;

const EMPTY: Rec = Rec {
    sym: Sym(0),
    bright: Visual(NONE),
    dark: Visual(NONE),
    over: Visual(NONE),
    pair: NONE,
    bidx: NONE,
    sidx: NONE,
    dimmed: Visual(NONE),
    factor: 0,
    flags: 0,
};

/// Color pairs by dense index, as their HLS.
#[derive(Default)]
struct Pairs {
    pairs: Vec<(Option<Hls>, Option<Hls>)>,
    colors: Vec<(Option<Color>, Option<Color>)>,
    index: HashMap<(u64, u64), u32, FxBuild>,
}

impl Pairs {
    fn intern(&mut self, fg: Option<Color>, bg: Option<Color>) -> u32 {
        let key = (fg.map_or(0, |c| c.color_arg.key()), bg.map_or(0, |c| c.color_arg.key()));
        *self.index.entry(key).or_insert_with(|| {
            self.pairs.push((fg.map(Hls::of), bg.map(Hls::of)));
            self.colors.push((fg, bg));
            self.pairs.len() as u32 - 1
        })
    }
}

/// Animation.adjust_color_brightness in two steps, float op for float op:
/// the color's hue, saturation and lightness once, then per brightness the
/// adjusted rgb.
#[derive(Clone, Copy)]
struct Hls {
    hue: f64,
    saturation: f64,
    lightness: f64,
}

impl Hls {
    fn of(color: Color) -> Hls {
        let (r, g, b) = color.rgb_ints();
        let normalized_red = r as f64 / 255.0;
        let normalized_green = g as f64 / 255.0;
        let normalized_blue = b as f64 / 255.0;
        let max_val = normalized_red.max(normalized_green).max(normalized_blue);
        let min_val = normalized_red.min(normalized_green).min(normalized_blue);
        let lightness = (max_val + min_val) / 2.0;
        let (hue, saturation) = if max_val == min_val {
            (0.0, 0.0)
        } else {
            let diff = max_val - min_val;
            let saturation =
                if lightness > 0.5 { diff / (2.0 - max_val - min_val) } else { diff / (max_val + min_val) };
            let hue = if max_val == normalized_red {
                (normalized_green - normalized_blue) / diff + if normalized_green < normalized_blue { 6.0 } else { 0.0 }
            } else if max_val == normalized_green {
                (normalized_blue - normalized_red) / diff + 2.0
            } else {
                (normalized_red - normalized_green) / diff + 4.0
            };
            (hue / 6.0, saturation)
        };
        Hls { hue, saturation, lightness }
    }

    /// The adjusted color as (present, rgb): see `unpack`.
    #[inline]
    fn adjust(&self, brightness: f64) -> u64 {
        fn hue_to_rgb(lightness_scaled: f64, color_intensity: f64, mut hue_value: f64) -> f64 {
            if hue_value < 0.0 {
                hue_value += 1.0;
            }
            if hue_value > 1.0 {
                hue_value -= 1.0;
            }
            if hue_value < 1.0 / 6.0 {
                return lightness_scaled + (color_intensity - lightness_scaled) * 6.0 * hue_value;
            }
            if hue_value < 1.0 / 2.0 {
                return color_intensity;
            }
            if hue_value < 2.0 / 3.0 {
                return lightness_scaled + (color_intensity - lightness_scaled) * (2.0 / 3.0 - hue_value) * 6.0;
            }
            lightness_scaled
        }
        let lightness = (self.lightness * brightness).min(1.0).max(0.0);
        let (red, green, blue) = if self.saturation == 0.0 {
            (lightness, lightness, lightness)
        } else {
            let color_intensity = if lightness < 0.5 {
                lightness * (1.0 + self.saturation)
            } else {
                lightness + self.saturation - lightness * self.saturation
            };
            let lightness_scaled = 2.0 * lightness - color_intensity;
            (
                hue_to_rgb(lightness_scaled, color_intensity, self.hue + 1.0 / 3.0),
                hue_to_rgb(lightness_scaled, color_intensity, self.hue),
                hue_to_rgb(lightness_scaled, color_intensity, self.hue - 1.0 / 3.0),
            )
        };
        let channel = |v: f64| round_half_even(v * 255.0) as u8 as u64;
        1 << 24 | channel(red) << 16 | channel(green) << 8 | channel(blue)
    }
}

#[derive(Clone, Copy, Default)]
struct MemoEntry<V> {
    wide: u64,
    /// The narrow key + 1 (0: empty).
    narrow: u32,
    value: V,
}

/// A (u32, u64) -> V memo, open addressing; emptied when half full.
struct Memo<V> {
    entries: Vec<MemoEntry<V>>,
    bits: u32,
    count: usize,
}

impl<V: Copy + Default> Memo<V> {
    fn new(bits: u32) -> Self {
        Memo { entries: vec![MemoEntry::default(); 1 << bits], bits, count: 0 }
    }

    /// The value, or Err(the empty entry to pass to `insert`).
    #[inline]
    fn get(&self, narrow: u32, wide: u64) -> Result<V, usize> {
        let narrow = narrow + 1;
        let h = ((narrow as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ wide).wrapping_mul(0xff51_afd7_ed55_8ccd);
        let mask = (1 << self.bits) - 1;
        let mut i = (h >> (64 - self.bits)) as usize;
        loop {
            let entry = self.entries.at(i);
            if entry.narrow == 0 {
                return Err(i);
            }
            if entry.narrow == narrow && entry.wide == wide {
                return Ok(entry.value);
            }
            i = (i + 1) & mask;
        }
    }

    fn clear(&mut self) {
        self.entries.fill(MemoEntry::default());
        self.count = 0;
    }

    #[inline]
    fn insert(&mut self, at: usize, narrow: u32, wide: u64, value: V) {
        if self.count >= self.entries.len() / 2 {
            // half full: start over (the entry found is not kept either)
            self.clear();
        } else {
            self.count += 1;
            *self.entries.at_mut(at) = MemoEntry { wide, narrow: narrow + 1, value };
        }
    }
}

/// A lit character: its number and input coordinate.
#[derive(Clone, Copy)]
struct Lit {
    at: u32,
    column: i32,
    row: i32,
}

/// A lit character's visual, or the lookup that finds it.
#[derive(Clone, Copy)]
enum Shine {
    Known(Visual),
    Dense(u32),
    Memo,
}

#[derive(Clone, Copy)]
struct Pending {
    /// The character's number (`Spotlights::order`).
    n: u32,
    shine: Shine,
    bits: u64,
    factor: f64,
}

/// A pair of generated colors (as adjust_color_brightness makes them) is
/// packed as (present, rgb) for fg in the low 25 bits and bg above.
fn unpack(packed: u64) -> (Option<Color>, Option<Color>) {
    let one = |v: u64| (v & 1 << 24 != 0).then(|| Color::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8));
    (one(packed & 0x1ff_ffff), one(packed >> 25))
}

pub struct Spotlights {
    config: SpotlightsConfig,
    recs: Vec<Rec>,
    /// The spotlightable characters are numbered in column order (the
    /// ellipses are walked a column at a time): their slots, and per
    /// number the stamp of the frame it was lit in.
    order: Vec<u32>,
    marks: Vec<u32>,
    /// The spotlightable character's number by input coordinate over the
    /// canvas cells, a column at a time (NONE elsewhere).
    grid: Vec<u32>,
    width: i64,
    height: i64,
    /// illuminated_chars, and the list being built this frame.
    lit: Vec<Lit>,
    next: Vec<Lit>,
    stamp: u32,
    spotlights: Vec<u32>,
    /// This frame's spotlight coordinates (live ones only).
    coords: Vec<Coord>,
    /// The last illumination's range and expand phase (its coordinates are
    /// `coords`), once there was one.
    lit_as: Option<(i64, bool)>,
    /// hypot(|dx|, 2 |dy|) by (|dy|, |dx|) below (height + 1, width + 1); 0:
    /// not computed yet.
    hyp: Vec<f64>,
    hyp_w: usize,
    pairs: Pairs,
    /// (bright visual, factor bits) -> dimmed visual.
    dimmed: Memo<u32>,
    /// (bright pair, factor bits) -> adjusted pair index.
    adjusted: Memo<u32>,
    /// The adjusted pairs: packed pair -> index, and back.
    pair_index: Memo<u32>,
    packed: Vec<u64>,
    /// (adjusted pair index, symbol index) -> visual (NONE: not yet), a row
    /// of `syms` per adjusted pair.
    vdense: Vec<Visual>,
    syms: usize,
    /// (symbol, packed adjusted pair) -> visual, for the other symbols.
    visuals: Memo<u32>,
    /// While searching (one range): factor bits -> dense factor index + 1,
    /// open addressing, and (factor index, bright index) -> dimmed visual
    /// (NONE: not yet), a row of `brights` per factor.
    fmap: Vec<(u64, u32)>,
    factors: usize,
    dense: Vec<Visual>,
    brights: usize,
    /// The ellipse's max_y_offset by |column offset| for `yoff_range`.
    yoff: Vec<i64>,
    yoff_range: i64,
    illuminate_range: i64,
    search_duration: i64,
    center: Name,
    searching: bool,
    expanding: bool,
    complete: bool,
    dynamic: bool,
    /// This frame's beam edge, falloff width and the squared integer distance
    /// below which a character is surely inside the edge.
    edge: f64,
    falloff_width: f64,
    core2: f64,
}

impl Spotlights {
    pub fn new(config: SpotlightsConfig) -> Self {
        Spotlights {
            config,
            recs: Vec::new(),
            marks: Vec::new(),
            order: Vec::new(),
            grid: Vec::new(),
            width: 0,
            height: 0,
            lit: Vec::new(),
            next: Vec::new(),
            stamp: 0,
            spotlights: Vec::new(),
            coords: Vec::new(),
            lit_as: None,
            hyp: Vec::new(),
            hyp_w: 0,
            pairs: Pairs::default(),
            dimmed: Memo::new(0),
            adjusted: Memo::new(0),
            pair_index: Memo::new(0),
            packed: Vec::new(),
            vdense: Vec::new(),
            syms: 0,
            visuals: Memo::new(0),
            fmap: Vec::new(),
            factors: 0,
            dense: Vec::new(),
            brights: 0,
            yoff: Vec::new(),
            yoff_range: -1,
            illuminate_range: 1,
            search_duration: 0,
            center: Name::NONE,
            searching: true,
            expanding: false,
            complete: false,
            dynamic: false,
            edge: 0.0,
            falloff_width: 0.0,
            core2: -1.0,
        }
    }

    /// SpotlightsIterator.make_spotlights, draw for draw.
    fn make_spotlights(&mut self, e: &mut Engine) -> Result<(), EngineError> {
        let config = self.config.clone();
        let minimum_distance = floor_div(e.canvas.right, 4) as f64;
        let names: Vec<Name> = (0..11).map(Name::auto).collect();
        for _ in 0..config.spotlight_count {
            let spawn_coord = e.canvas.random_coord(&mut e.rng, true, false);
            let spotlight = e.add_character("O", spawn_coord);
            self.spotlights.push(spotlight);

            let mut targets = [Coord::new(0, 0); 11];
            targets[0] = e.canvas.random_coord(&mut e.rng, false, false);
            for i in 1..11 {
                // find_coord_at_minimum_distance
                targets[i] = loop {
                    let coord = e.canvas.random_coord(&mut e.rng, false, false);
                    if geometry::find_length_of_line(targets[i - 1], coord, false) >= minimum_distance {
                        break coord;
                    }
                };
            }
            for (i, &target) in targets.iter().enumerate() {
                let speed = e.rng.uniform(config.search_speed_range.0, config.search_speed_range.1);
                let path = e.path_new(spotlight, speed, Some(Easing::InOutQuad), None, 0, false, names[i]).map_err(other)?;
                let control = e.canvas.random_coord(&mut e.rng, true, false);
                e.path_new_waypoint(path, target, Some(&[control]), Name::NONE).map_err(other)?;
            }
            e.chain_paths(spotlight, &names, true).map_err(other)?;

            let center = e.canvas.center;
            let path = e.path_new(spotlight, 0.5, Some(Easing::InOutSine), None, 0, false, self.center).map_err(other)?;
            e.path_new_waypoint(path, center, None, Name::NONE).map_err(other)?;
        }
        Ok(())
    }

    /// The ellipse's max_y_offset for every |column offset| of this range
    /// (coords_in_circle's, which is symmetric in the offset).
    fn ellipse(&mut self, range: i64) {
        if self.yoff_range == range {
            return;
        }
        self.yoff_range = range;
        self.yoff.clear();
        let a_squared = (range as f64).powf(2.0);
        let b_squared = (range as f64 / 2.0).powf(2.0);
        for dx in 0..=range {
            let x_component = (dx as f64).powf(2.0) / a_squared;
            self.yoff.push((b_squared * (1.0 - x_component)).powf(0.5) as i64);
        }
    }

    /// The smallest find_length_of_line(spotlight, input coordinate, true)
    /// over the live spotlights, folded from +inf with f64::min.
    #[inline]
    fn distance(&mut self, input: Coord) -> f64 {
        let mut min = f64::INFINITY;
        for &c in &self.coords {
            let dx = (input.column - c.column).unsigned_abs() as usize;
            let dy = (input.row - c.row).unsigned_abs() as usize;
            let d = if dx < self.hyp_w && dy <= self.height as usize {
                let entry = self.hyp.at_mut(dy * self.hyp_w + dx);
                if *entry == 0.0 {
                    *entry = f64::hypot(dx as f64, 2.0 * dy as f64);
                }
                *entry
            } else {
                f64::hypot(dx as f64, 2.0 * dy as f64)
            };
            min = min.min(d);
        }
        min
    }

    /// The visual of an illuminated character: its bright pair, dimmed past
    /// the beam's edge by max(1 - (distance - edge) / falloff width, 0.2);
    /// or, for a dimmed visual not known yet, the lookup to make.
    #[inline(always)]
    fn prepare(&mut self, n: u32, input: Coord) -> Pending {
        let known = |visual| Pending { n, shine: Shine::Known(visual), bits: 0, factor: 0.0 };
        let rec = *self.recs.at(n);
        if self.expanding && rec.over.0 != NONE {
            return known(rec.over);
        }
        if rec.flags & FIXED != 0 {
            return known(rec.bright);
        }
        let mut nearest = i64::MAX;
        for &c in &self.coords {
            let dx = input.column - c.column;
            let dy = 2 * (input.row - c.row);
            nearest = nearest.min(dx * dx + dy * dy);
        }
        if (nearest as f64) < self.core2 {
            return known(rec.bright);
        }
        let distance = self.distance(input);
        if distance <= self.edge {
            return known(rec.bright);
        }
        let factor = (1.0 - (distance - self.edge) / self.falloff_width).max(0.2);
        let bits = factor.to_bits();
        if rec.factor == bits && rec.dimmed.0 != NONE {
            return known(rec.dimmed);
        }
        self.recs.at_mut(n).factor = bits;
        let shine = if rec.bidx != NONE && !self.expanding {
            self.dense_index(bits, rec.bidx).map_or(Shine::Memo, Shine::Dense)
        } else {
            Shine::Memo
        };
        Pending { n, shine, bits, factor }
    }

    #[inline(always)]
    fn resolve(&mut self, e: &mut Engine, p: Pending) -> Visual {
        let visual = match p.shine {
            Shine::Known(visual) => return visual,
            Shine::Dense(at) => {
                let visual = *self.dense.at(at);
                if visual.0 != NONE {
                    visual
                } else {
                    let visual = self.adjusted_visual(e, p.n, p.bits, p.factor);
                    *self.dense.at_mut(at) = visual;
                    visual
                }
            }
            Shine::Memo => self.dimmed(e, p.n, p.bits, p.factor),
        };
        self.recs.at_mut(p.n).dimmed = visual;
        visual
    }

    /// Memoized by (bright visual, factor), else `adjusted_visual`.
    fn dimmed(&mut self, e: &mut Engine, n: u32, bits: u64, factor: f64) -> Visual {
        let bright = self.recs.at(n).bright;
        let at = match self.dimmed.get(bright.0, bits) {
            Ok(visual) => return Visual(visual),
            Err(at) => at,
        };
        let visual = self.adjusted_visual(e, n, bits, factor);
        self.dimmed.insert(at, bright.0, bits, visual.0);
        visual
    }

    /// The index of (factor, bright index) in `dense`, adding a new factor's
    /// row; None when the table is full.
    #[inline]
    fn dense_index(&mut self, bits: u64, bidx: u32) -> Option<u32> {
        let mask = (1 << FMAP_BITS) - 1;
        let mut i = (bits.wrapping_mul(0xff51_afd7_ed55_8ccd) >> (64 - FMAP_BITS)) as usize;
        let fidx = loop {
            let (key, fidx) = *self.fmap.at(i);
            if fidx == 0 {
                if self.factors == DENSE_FACTORS {
                    return None;
                }
                self.factors += 1;
                *self.fmap.at_mut(i) = (bits, self.factors as u32);
                self.dense.resize(self.factors * self.brights, Visual(NONE));
                break self.factors - 1;
            }
            if key == bits {
                break fidx as usize - 1;
            }
            i = (i + 1) & mask;
        };
        Some((fidx * self.brights + bidx as usize) as u32)
    }

    /// The adjusted visual through (bright pair, factor) -> adjusted pair and
    /// (adjusted pair, symbol) -> visual.
    fn adjusted_visual(&mut self, e: &mut Engine, n: u32, bits: u64, factor: f64) -> Visual {
        let rec = *self.recs.at(n);
        let index = match self.adjusted.get(rec.pair, bits) {
            Ok(index) => index,
            Err(at) => {
                let (fg, bg) = self.pairs.pairs[rec.pair as usize];
                let fg = fg.map_or(0, |c| c.adjust(factor));
                let bg = bg.map_or(0, |c| c.adjust(factor));
                let packed = fg | bg << 25;
                debug_assert_eq!(unpack(packed), {
                    let (fg, bg) = self.pairs.colors[rec.pair as usize];
                    let adjust = |c: Color| Animation::adjust_color_brightness(&c, factor);
                    (fg.map(adjust), bg.map(adjust))
                });
                let index = match self.pair_index.get(0, packed) {
                    Ok(index) => index,
                    Err(_) if self.packed.len() == ADJUSTED_PAIRS => {
                        self.pair_index.clear();
                        self.packed.clear();
                        self.vdense.clear();
                        self.adjusted.clear();
                        return self.adjusted_visual(e, n, bits, factor);
                    }
                    Err(at) => {
                        let index = self.packed.len() as u32;
                        self.packed.push(packed);
                        let len = self.vdense.len();
                        // inside the capacity (reserved for every pair)
                        assert!(len + self.syms <= self.vdense.capacity());
                        self.vdense.resize(len + self.syms, Visual(NONE));
                        self.pair_index.insert(at, 0, packed, index);
                        index
                    }
                };
                self.adjusted.insert(at, rec.pair, bits, index);
                index
            }
        };
        if rec.sidx != NONE {
            let at = index as usize * self.syms + rec.sidx as usize;
            let visual = *self.vdense.at(at);
            if visual.0 != NONE {
                return visual;
            }
            let visual = self.make_visual(e, n, index);
            *self.vdense.at_mut(at) = visual;
            return visual;
        }
        let sym = rec.sym;
        let packed = *self.packed.at(index);
        match self.visuals.get(sym.0, packed) {
            Ok(visual) => Visual(visual),
            Err(at) => {
                let visual = self.make_visual(e, n, index);
                self.visuals.insert(at, sym.0, packed, visual.0);
                visual
            }
        }
    }

    fn make_visual(&self, e: &mut Engine, n: u32, index: u32) -> Visual {
        let (fg, bg) = unpack(*self.packed.at(index));
        e.visuals.make(&e.symbols, VisualInfo { sym: self.recs.at(n).sym, fg, bg, attrs: HAS_COLORS })
    }

    /// SpotlightsIterator.illuminate_chars(illuminate_range), over the canvas
    /// cells only. Characters met in an ellipse are stamped and listed once a
    /// frame and given their colors; previously lit ones without this frame's
    /// stamp go dark.
    fn illuminate(&mut self, e: &mut Engine) {
        let range = self.illuminate_range;
        let override_on = self.expanding;
        // the lit set and visuals depend only on the live spotlights'
        // coordinates, the range and the expand phase; when none changed
        // (spotlights move under a cell a frame) nothing would change
        let same = self.lit_as == Some((range, override_on))
            && self.coords.len() == self.spotlights.len()
            && self.spotlights.iter().zip(&self.coords).all(|(&s, &c)| e.coord(s) == c);
        if same {
            return;
        }
        self.lit_as = Some((range, override_on));
        if self.yoff_range != range {
            // factors of another range don't come back
            self.adjusted.clear();
        }
        self.ellipse(range);
        self.stamp += 1;
        let stamp = self.stamp;
        self.next.clear();
        self.coords.clear();
        for &s in &self.spotlights {
            self.coords.push(e.coord(s));
        }
        let (width, height) = (self.width, self.height);
        for k in 0..self.coords.len() {
            let Coord { column: h, row: v } = self.coords[k];
            let first = (h - range).max(1);
            let last = (h + range).min(width);
            for x in first..=last {
                let off = *self.yoff.at((x - h).unsigned_abs() as usize);
                let lo = (v - off).max(1);
                let hi = (v + off).min(height);
                let mut cell = ((x - 1) * height + (lo - 1)) as usize;
                for y in lo..=hi {
                    let at = *self.grid.at(cell);
                    cell += 1;
                    if at == NONE {
                        continue;
                    }
                    let mark = self.marks.at_mut(at);
                    if *mark == stamp {
                        continue;
                    }
                    *mark = stamp;
                    self.next.push(Lit { at, column: x as i32, row: y as i32 });
                }
            }
        }
        for i in 0..self.next.len() {
            let Lit { at, column, row } = *self.next.at(i);
            let p = self.prepare(at, Coord::new(column as i64, row as i64));
            let visual = self.resolve(e, p);
            e.set_visual(*self.order.at(at), visual);
        }
        for &Lit { at, .. } in &self.lit {
            if *self.marks.at(at) == stamp {
                continue;
            }
            let rec = self.recs.at(at);
            let visual = if override_on && rec.over.0 != NONE { rec.over } else { rec.dark };
            e.set_visual(*self.order.at(at), visual);
        }
        std::mem::swap(&mut self.lit, &mut self.next);
    }
}

impl Hooks for Spotlights {}

fn other(message: String) -> EngineError {
    EngineError::Other(message)
}

/// Animation.set_appearance(input symbol, colors)'s visual: under
/// --existing-color-handling always, a character using its input colors
/// shows those and its bold instead.
fn appearance(e: &mut Engine, slot: u32, fg: Option<Color>, bg: Option<Color>) -> Visual {
    let sym = e.input_sym(slot);
    let info = if e.existing_color_handling() == ExistingColorHandling::Always && e.uses_preexisting_colors(slot) {
        let attrs = HAS_COLORS | if e.input_bold(slot) { BOLD } else { 0 };
        VisualInfo { sym, fg: e.input_fg(slot), bg: e.input_bg(slot), attrs }
    } else {
        VisualInfo { sym, fg, bg, attrs: HAS_COLORS }
    };
    e.visuals.make(&e.symbols, info)
}

/// The visual of _adjust_color_pair_brightness(pair, brightness).
fn adjusted(e: &mut Engine, slot: u32, fg: Option<Color>, bg: Option<Color>, brightness: f64) -> Visual {
    let fg = fg.map(|c| Animation::adjust_color_brightness(&c, brightness));
    let bg = bg.map(|c| Animation::adjust_color_brightness(&c, brightness));
    appearance(e, slot, fg, bg)
}

impl Effect for Spotlights {
    fn build(&mut self, e: &mut Engine) -> Result<(), EngineError> {
        let config = self.config.clone();
        self.center = e.name("center");
        self.make_spotlights(e)?;
        let canvas = e.canvas.clone();
        let final_gradient =
            Gradient::new(&config.final_gradient_stops, &config.final_gradient_steps, false, false).map_err(other)?;
        let final_gradient_mapping = final_gradient
            .build_coordinate_color_mapping(
                canvas.text_bottom,
                canvas.text_top,
                canvas.text_left,
                canvas.text_right,
                config.final_gradient_direction,
            )
            .map_err(other)?;
        // SpotlightsIterator.DYNAMIC_NEUTRAL_GRAY
        let gray = Color::from_hex("#808080").unwrap();
        let dynamic = e.existing_color_handling() == ExistingColorHandling::Dynamic;
        let always = e.existing_color_handling() == ExistingColorHandling::Always;
        self.dynamic = dynamic;
        let space = e.sym(" ");
        self.width = canvas.right.max(0);
        self.height = canvas.top.max(0);
        // by slot, and the slot by input coordinate a row at a time, first
        let mut recs = vec![EMPTY; e.char_count()];
        let mut by_row = vec![NONE; (self.width * self.height) as usize];
        let characters = e.get_characters(CharacterFilter::default(), CharacterSort::TopToBottomLeftToRight);
        let mut brights: HashMap<Visual, u32, FxBuild> = HashMap::default();
        let mut syms: HashMap<Sym, u32, FxBuild> = HashMap::default();
        self.lit = Vec::with_capacity(characters.len());
        self.next = Vec::with_capacity(characters.len());
        for &slot in &characters {
            let input = e.input_coord(slot);
            let (input_fg, input_bg) = (e.input_fg(slot), e.input_bg(slot));
            let mut flags = 0;
            if e.input_sym(slot) != space || input_fg.is_some() || input_bg.is_some() {
                flags |= LIT;
            }
            if always && e.uses_preexisting_colors(slot) {
                flags |= FIXED;
            }
            let (fg, bg) = if dynamic {
                (Some(input_fg.unwrap_or(gray)), input_bg)
            } else {
                (Some(*final_gradient_mapping.get(&input).unwrap()), None)
            };
            let bright = appearance(e, slot, fg, bg);
            let dark = adjusted(e, slot, fg, bg, 0.2);
            let pair = self.pairs.intern(fg, bg);
            // _get_expand_color_override: (None, bg) for a bg-only character,
            // (None, None) for one without input colors
            let over = if dynamic && input_fg.is_none() { appearance(e, slot, None, input_bg) } else { Visual(NONE) };
            let bidx = if flags & FIXED != 0 {
                NONE
            } else {
                let next = brights.len() as u32;
                let bidx = *brights.entry(bright).or_insert(next);
                if bidx as usize >= DENSE_BRIGHTS {
                    NONE
                } else {
                    bidx
                }
            };
            let sidx = if flags & FIXED != 0 {
                NONE
            } else {
                let next = syms.len() as u32;
                let sidx = *syms.entry(e.input_sym(slot)).or_insert(next);
                if sidx as usize >= DENSE_SYMS {
                    NONE
                } else {
                    sidx
                }
            };
            recs[slot as usize] = Rec { sym: e.input_sym(slot), bright, dark, over, pair, bidx, sidx, flags, ..EMPTY };
            if flags & LIT != 0 {
                by_row[((input.row - 1) * self.width + (input.column - 1)) as usize] = slot;
            }
            e.set_visible(slot, true);
            e.set_visual(slot, dark);
        }
        let (width, height) = (self.width as usize, self.height as usize);
        self.grid = vec![NONE; width * height];
        for x in 0..width {
            for y in 0..height {
                let slot = by_row[y * width + x];
                if slot != NONE {
                    self.grid[x * height + y] = self.order.len() as u32;
                    self.order.push(slot);
                    self.recs.push(recs[slot as usize]);
                }
            }
        }
        self.marks = vec![0; self.order.len()];
        self.brights = brights.len().min(DENSE_BRIGHTS);
        self.fmap = vec![(0, 0); 1 << FMAP_BITS];
        self.factors = 0;
        self.dense = Vec::with_capacity(DENSE_FACTORS * self.brights);
        self.hyp_w = self.width as usize + 1;
        self.hyp = vec![0.0; self.hyp_w * (self.height as usize + 1)];
        // room for ~64 factors of every bright visual, and of every pair
        let bits = |keys: usize| (keys * 128).next_power_of_two().trailing_zeros();
        self.dimmed = Memo::new(bits(brights.len()).clamp(16, 19));
        self.adjusted = Memo::new(bits(self.pairs.pairs.len()).clamp(12, 18));
        self.pair_index = Memo::new(ADJUSTED_BITS);
        self.packed = Vec::with_capacity(ADJUSTED_PAIRS);
        self.syms = syms.len().min(DENSE_SYMS);
        self.vdense = Vec::with_capacity(ADJUSTED_PAIRS * self.syms);
        self.visuals = Memo::new(if syms.len() > DENSE_SYMS { 18 } else { 1 });
        // a run makes ~16 dimmed visuals per character
        e.visuals.reserve((e.char_count() * 16).min(200_000), 24);
        let smallest = canvas.right.min(canvas.top);
        // int(min(smallest // ratio, smallest)) - float floor division then
        // truncation
        self.illuminate_range =
            ((smallest as f64 / config.beam_width_ratio).floor().min(smallest as f64) as i64).max(1);
        let largest = canvas.right.max(canvas.top);
        self.yoff.reserve((largest as f64 / 1.5).max(0.0) as usize + self.illuminate_range as usize + 2);
        self.search_duration = config.search_duration;
        self.searching = true;
        self.expanding = false;
        self.complete = false;
        let first = Name::auto(0);
        for i in 0..self.spotlights.len() {
            let spotlight = self.spotlights[i];
            e.activate_path_name(self, spotlight, first);
            e.active_insert(spotlight);
        }
        self.coords.reserve(self.spotlights.len());
        Ok(())
    }

    fn next_frame(&mut self, e: &mut Engine) -> bool {
        if self.complete {
            return false;
        }
        let range = self.illuminate_range as f64;
        let falloff = self.config.beam_falloff;
        self.edge = range * (1.0 - falloff);
        self.falloff_width = range * falloff;
        // edge^2 less a relative 1e-9 (far above hypot's error), or nothing
        self.core2 = if self.edge > 0.0 { self.edge * self.edge * 0.999_999_999 } else { -1.0 };
        self.illuminate(e);
        if self.searching {
            self.search_duration -= 1;
            if self.search_duration == 0 {
                let center = self.center;
                for i in 0..self.spotlights.len() {
                    let spotlight = self.spotlights[i];
                    e.activate_path_name(self, spotlight, center);
                }
                self.searching = false;
            }
        }
        if !self.spotlights.iter().any(|&s| e.ch.path[s as usize] != NONE) {
            self.spotlights.truncate(1);
            self.expanding = true;
            self.illuminate_range += 1;
            let limit = (e.canvas.right.max(e.canvas.top) as f64 / 1.5).floor();
            if self.illuminate_range as f64 > limit {
                self.complete = true;
            }
        }
        e.update(self);
        true
    }
}
