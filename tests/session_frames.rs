//! The wasm `Session`'s engine path, exercised natively: default effect config,
//! `fx::Engine` + stepping + `pack_display_frame`. Whatever the Session does
//! in the browser must hold here first — same calls, same order.

use ttfx::effects::EffectCommand;
use ttfx::engine::animation::ExistingColorHandling;
use ttfx::engine::canvas::Anchor;
use ttfx::engine::ctx::Clock;
use ttfx::engine::terminal::TerminalConfig;
use ttfx::fx::run::Effect;
use ttfx::fx::{Engine, PackedFrame};
use ttfx::utils::graphics::Color;
use ttfx::utils::palette::Palette;
use ttfx::utils::rng::Rng;

/// Mirror of `Session::new` + stepping: run to completion, collect frames.
fn run_packed(
    input: &str,
    name: &str,
    seed: u64,
    palette: Option<&Palette>,
) -> Vec<PackedFrame> {
    let mut command = EffectCommand::with_defaults(name)
        .unwrap_or_else(|| panic!("unknown effect '{name}'"));
    if let Some(palette) = palette {
        command.apply_palette(palette, &Default::default());
    }
    let config = TerminalConfig {
        frame_rate: 60,
        canvas_width: 0,
        canvas_height: 0,
        anchor_canvas: Anchor::C,
        anchor_text: Anchor::C,
        reuse_canvas: true,
        no_eol: true,
        no_restore_cursor: true,
        terminal_background_color: Color::from_hex("000000").unwrap(),
        terminal_size: Some((40, 12)),
        existing_color_handling: ExistingColorHandling::Ignore,
        ..TerminalConfig::default()
    };
    let clock = Clock::virtual_with_frame_rate(60);
    let mut engine = Engine::new(input, config, Rng::seeded(seed), clock).unwrap();
    let Some(mut effect) = ttfx::fx::effects::build(&command) else {
        panic!("fx declined effect");
    };
    effect
        .build(&mut engine)
        .map_err(|e| e.to_string())
        .expect("effect build failed");
    let mut frames = Vec::new();
    for _ in 0..100_000 {
        if !effect.next_frame(&mut engine) {
            break;
        }
        frames.push(engine.pack_display_frame());
    }
    assert!(!frames.is_empty(), "{name} produced no frames");
    frames
}

fn assert_consistent(frame: &PackedFrame) {
    let n = frame.width * frame.height;
    assert!(frame.width > 0 && frame.height > 0);
    assert_eq!(frame.symbols.len(), n, "symbols truncated");
    assert_eq!(frame.fg.len(), n, "fg truncated");
    assert_eq!(frame.bg.len(), n, "bg truncated");
    assert_eq!(frame.flags.len(), n, "flags truncated");
}

/// Visible (non-hidden) text of the final frame.
fn visible_text(frame: &PackedFrame) -> String {
    frame
        .symbols
        .iter()
        .zip(frame.flags.iter())
        .filter(|(_, f)| **f & PackedFrame::HIDDEN == 0)
        .map(|(s, _)| char::from_u32(*s).unwrap_or('�'))
        .collect()
}

#[test]
fn session_path_terminates_with_consistent_frames() {
    // Resolving effects must end on the input text; ambient ones (matrix,
    // thunderstorm) only have to terminate with well-formed frames.
    for name in ["decrypt", "beams", "wipe", "rain", "matrix"] {
        let frames = run_packed("hi", name, 1, None);
        for frame in &frames {
            assert_consistent(frame);
        }
        if name == "matrix" {
            continue;
        }
        let last = visible_text(frames.last().unwrap());
        assert!(
            last.contains("hi"),
            "{name} final frame lost the input: {last:?}"
        );
    }
}

#[test]
fn session_path_is_deterministic_per_seed() {
    let a = run_packed("hello", "decrypt", 42, None);
    let b = run_packed("hello", "decrypt", 42, None);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.symbols, y.symbols);
        assert_eq!(x.fg, y.fg);
    }
    let c = run_packed("hello", "decrypt", 43, None);
    // Frame 0 can predate the first draw; the full streams must diverge.
    let same = a.len() == c.len()
        && a
            .iter()
            .zip(c.iter())
            .all(|(x, y)| x.symbols == y.symbols && x.fg == y.fg);
    assert!(!same, "different seeds must diverge");
}

#[test]
fn session_path_palette_recolors() {
    let palette =
        Palette::from_hex_list("#daecc6,#bbdd97,#9ece6a,#678549,#39482e").unwrap();
    let frames = run_packed("OMARCHY", "decrypt", 1, Some(&palette));
    let last = frames.last().unwrap();
    assert_consistent(last);
    let allowed = [
        0xdaecc6u32, 0xbbdd97, 0x9ece6a, 0x678549, 0x39482e,
    ];
    let mut painted = 0;
    for (s, f) in last.symbols.iter().zip(last.fg.iter()) {
        let ch = char::from_u32(*s).unwrap_or(' ');
        if ch != ' ' && ch != '\0' {
            painted += 1;
            assert!(
                allowed.contains(&(f & 0xffffff)),
                "fg {f:#x} outside the palette"
            );
        }
    }
    assert!(painted > 0, "palette run painted nothing");
}
