//! The visual pool: every CharacterVisual is formatted once (SGR prefix,
//! symbol, reset) and interned. A visual is then a `Visual` handle, and the
//! renderer only copies bytes: it never formats, allocates or refcounts.
//!
//! The key is what the visual *is* (symbol, colors, attributes); the bytes are
//! a function of it and of the run's color flags, which are fixed.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::engine::animation::CharacterVisual;
use crate::utils::{ansi, hexterm};
use crate::utils::graphics::{Color, ColorPair};

use super::{FxBuild, Sym, Symbols};

/// A pooled visual.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Visual(pub u32);

// Attribute bits, in format_symbol_into's order. `dim` is stored but never
// emitted, faithfully.
pub const BOLD: u16 = 1;
pub const ITALIC: u16 = 2;
pub const UNDERLINE: u16 = 4;
pub const BLINK: u16 = 8;
pub const REVERSE: u16 = 16;
pub const HIDDEN: u16 = 32;
pub const STRIKE: u16 = 64;
pub const DIM: u16 = 128;
/// `colors` is `Some(pair)` (possibly both None) rather than `None`.
pub const HAS_COLORS: u16 = 256;

/// What a visual is: CharacterVisual's logical fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisualInfo {
    pub sym: Sym,
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub attrs: u16,
}

/// One word per color, 0 when absent: equal exactly when the colors are.
#[inline]
pub fn color_key(color: Option<Color>) -> u64 {
    color.map_or(0, |c| c.color_arg.key())
}

/// A VisualInfo packed into three words, equal exactly when the infos are:
/// symbol, attributes and which colors are present, then each color's key
/// (0 when absent). The pool's map holds these rather than the infos, so a
/// probe compares 24 bytes in a table half the size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key([u64; 3]);

impl Key {
    #[inline]
    fn of(info: &VisualInfo) -> Key {
        let present = (info.fg.is_some() as u64) << 48 | (info.bg.is_some() as u64) << 49;
        Key([
            info.sym.0 as u64 | (info.attrs as u64) << 32 | present,
            info.fg.map_or(0, |c| c.color_arg.key()),
            info.bg.map_or(0, |c| c.color_arg.key()),
        ])
    }
}

impl Hash for Key {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0[0]);
        state.write_u64(self.0[1]);
        state.write_u64(self.0[2]);
    }
}

impl VisualInfo {
    /// CharacterVisual.colors.
    pub fn colors(&self) -> Option<ColorPair> {
        (self.attrs & HAS_COLORS != 0).then(|| ColorPair::new(self.fg, self.bg))
    }
}

/// Byte span of a visual in the pool.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub offset: u32,
    pub len: u32,
}

/// Every visual is readable this far past its start, so emission may copy a
/// fixed block and then advance by the real length.
pub const COPY_BLOCK: usize = 64;

pub struct VisualPool {
    no_color: bool,
    xterm_colors: bool,
    map: HashMap<Key, Visual, FxBuild>,
    pub info: Vec<VisualInfo>,
    pub spans: Vec<Span>,
    /// Formatted bytes of every visual, plus COPY_BLOCK bytes of slack.
    pub bytes: Vec<u8>,
    /// The longest visual's byte length.
    pub max_len: usize,
}

impl VisualPool {
    pub fn new(no_color: bool, xterm_colors: bool) -> Self {
        VisualPool {
            no_color,
            xterm_colors,
            // effects with gradients make tens of thousands of visuals: no
            // rehash until then (the table's entries are touched only as used)
            map: HashMap::with_capacity_and_hasher(1 << 14, FxBuild::default()),
            info: Vec::with_capacity(1 << 14),
            spans: Vec::with_capacity(1 << 14),
            bytes: vec![0; COPY_BLOCK],
            max_len: 1,
        }
    }

    /// The visual for these fields, formatting it on first use.
    pub fn make(&mut self, symbols: &Symbols, info: VisualInfo) -> Visual {
        let VisualPool { map, no_color, xterm_colors, .. } = self;
        let entry = match map.entry(Key::of(&info)) {
            Entry::Occupied(o) => return *o.get(),
            Entry::Vacant(v) => v,
        };
        let (no_color, xterm_colors) = (*no_color, *xterm_colors);
        let offset = self.bytes.len() - COPY_BLOCK;
        self.bytes.truncate(offset);
        format_visual(&mut self.bytes, symbols.get(info.sym), &info, no_color, xterm_colors);
        let len = self.bytes.len() - offset;
        self.bytes.resize(self.bytes.len() + COPY_BLOCK, 0);
        self.max_len = self.max_len.max(len);
        let handle = Visual(self.info.len() as u32);
        self.info.push(info);
        self.spans.push(Span { offset: offset as u32, len: len as u32 });
        *entry.insert(handle)
    }

    /// Room for `additional` more visuals (of about `bytes` bytes each)
    /// without growing the tables.
    pub fn reserve(&mut self, additional: usize, bytes: usize) {
        self.map.reserve(additional);
        self.info.reserve(additional);
        self.spans.reserve(additional);
        self.bytes.reserve(additional * bytes);
    }

    /// A visual that is just these bytes, outside the map (the renderer's
    /// blank cell).
    pub fn raw(&mut self, sym: Sym, bytes: &[u8]) -> Visual {
        let offset = self.begin();
        self.bytes.extend_from_slice(bytes);
        self.finish(VisualInfo { sym, fg: None, bg: None, attrs: 0 }, offset)
    }

    /// Start a visual's bytes where the slack begins.
    fn begin(&mut self) -> usize {
        let offset = self.bytes.len() - COPY_BLOCK;
        self.bytes.truncate(offset);
        offset
    }

    /// The bytes from `offset` on are the visual's: restore the slack.
    fn finish(&mut self, info: VisualInfo, offset: usize) -> Visual {
        let len = self.bytes.len() - offset;
        self.bytes.resize(self.bytes.len() + COPY_BLOCK, 0);
        self.max_len = self.max_len.max(len);
        let handle = Visual(self.info.len() as u32);
        self.info.push(info);
        self.spans.push(Span { offset: offset as u32, len: len as u32 });
        handle
    }

    #[inline]
    pub fn info(&self, visual: Visual) -> &VisualInfo {
        &self.info[visual.0 as usize]
    }

    #[inline]
    pub fn bytes_of(&self, visual: Visual) -> &[u8] {
        let span = self.spans[visual.0 as usize];
        &self.bytes[span.offset as usize..(span.offset + span.len) as usize]
    }
}

/// Converts the old engine's CharacterVisual (input parsing can set
/// appearances) into a pool key.
pub fn info_of(symbols: &mut Symbols, visual: &CharacterVisual) -> VisualInfo {
    let mut attrs = 0;
    for (on, bit) in [
        (visual.bold, BOLD),
        (visual.italic, ITALIC),
        (visual.underline, UNDERLINE),
        (visual.blink, BLINK),
        (visual.reverse, REVERSE),
        (visual.hidden, HIDDEN),
        (visual.strike, STRIKE),
        (visual.dim, DIM),
    ] {
        if on {
            attrs |= bit;
        }
    }
    let (fg, bg) = match visual.colors {
        Some(pair) => {
            attrs |= HAS_COLORS;
            (pair.fg_color, pair.bg_color)
        }
        None => (None, None),
    };
    VisualInfo { sym: symbols.intern(&visual.symbol), fg, bg, attrs }
}

/// CharacterVisual.format_symbol_into for these fields: the SGR attributes
/// in upstream's fixed order (`dim` intentionally omitted), the colors, the
/// symbol, and a reset when anything came before it.
fn format_visual(out: &mut Vec<u8>, symbol: &str, info: &VisualInfo, no_color: bool, xterm: bool) {
    // the attributes and colors, at most 7 * 5 + 2 * 19 bytes, go through a
    // buffer on the stack
    let mut sgr = Sgr { buf: [0; 80], len: 0 };
    for (bit, code) in [
        (BOLD, ansi::BOLD),
        (ITALIC, ansi::ITALIC),
        (UNDERLINE, ansi::UNDERLINE),
        (BLINK, ansi::BLINK),
        (REVERSE, ansi::REVERSE),
        (HIDDEN, ansi::HIDDEN),
        (STRIKE, ansi::STRIKETHROUGH),
    ] {
        if info.attrs & bit != 0 {
            sgr.put_str(code);
        }
    }
    if !no_color {
        if let Some(fg) = &info.fg {
            sgr.color(fg, b"38", xterm);
        }
        if let Some(bg) = &info.bg {
            sgr.color(bg, b"48", xterm);
        }
    }
    out.extend_from_slice(&sgr.buf[..sgr.len]);
    out.extend_from_slice(symbol.as_bytes());
    if sgr.len != 0 {
        out.extend_from_slice(ansi::RESET_ALL.as_bytes());
    }
}

/// A visual's SGR sequences before its symbol.
struct Sgr {
    buf: [u8; 80],
    len: usize,
}

impl Sgr {
    /// Fixed-size copies (a copy of a length only known at run time would
    /// be a memcpy call).
    #[inline(always)]
    fn put<const N: usize>(&mut self, bytes: &[u8; N]) {
        self.buf[self.len..self.len + N].copy_from_slice(bytes);
        self.len += N;
    }

    fn put_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.buf[self.len] = b;
            self.len += 1;
        }
    }

    /// resolve_color_code + colorterm._color: `ESC[38;2;r;g;bm` from the
    /// hex string, or `ESC[38;5;nm` under --xterm-colors.
    #[inline(always)]
    fn color(&mut self, color: &Color, location: &[u8; 2], xterm: bool) {
        self.put(b"\x1b[");
        self.put(location);
        if xterm {
            let code = color.xterm_color.unwrap_or_else(|| hexterm::hex_to_xterm(&color.rgb_color));
            self.put(b";5;");
            self.decimal(code);
        } else {
            // the channels of the hex string (Color parses them once)
            let (r, g, b) = color.rgb_ints();
            self.put(b";2;");
            self.decimal(r);
            self.put(b";");
            self.decimal(g);
            self.put(b";");
            self.decimal(b);
        }
        self.put(b"m");
    }

    /// The value's decimal digits: three bytes from a table, of which the
    /// first `n` count (the rest is written over next).
    #[inline(always)]
    fn decimal(&mut self, value: u8) {
        let (digits, n) = DECIMAL[value as usize];
        self.put(&digits);
        self.len -= 3 - n as usize;
    }
}

/// Every u8's decimal digits, left-aligned in three bytes, and their count.
static DECIMAL: [([u8; 3], u8); 256] = {
    let mut t = [([0u8; 3], 0u8); 256];
    let mut v = 0;
    while v < 256 {
        let (h, d, u) = (b'0' + (v / 100) as u8, b'0' + (v / 10 % 10) as u8, b'0' + (v % 10) as u8);
        t[v] = if v >= 100 {
            ([h, d, u], 3)
        } else if v >= 10 {
            ([d, u, 0], 2)
        } else {
            ([u, 0, 0], 1)
        };
        v += 1;
    }
    t
};
