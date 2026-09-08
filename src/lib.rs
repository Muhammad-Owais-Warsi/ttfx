pub mod cli;
pub mod effects;
pub mod engine;
pub mod utils;

use std::sync::atomic::{AtomicBool, Ordering};

/// `println!` and `eprintln!` panic when their write fails, and a release
/// build aborts on panic — so on a terminal that has just gone away, reporting
/// the loss is what dumps the core, not the loss itself:
///
/// ```text
/// thread 'main' panicked at library/std/src/io/stdio.rs:1166:9:
/// failed printing to stderr: Input/output error (os error 5)
/// ```
///
/// Nothing ttfx says is worth dying over, so messages go out through these two
/// and a failed write is dropped (basecamp/omarchy#6762).
#[macro_export]
macro_rules! outln {
    ($($arg:tt)*) => {{
        let _ = ::std::io::Write::write_fmt(
            &mut ::std::io::stdout(),
            format_args!("{}\n", format_args!($($arg)*)),
        );
    }};
}

/// [`outln!`] for stderr.
#[macro_export]
macro_rules! errln {
    ($($arg:tt)*) => {{
        let _ = ::std::io::Write::write_fmt(
            &mut ::std::io::stderr(),
            format_args!("{}\n", format_args!($($arg)*)),
        );
    }};
}

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static TERMINATED: AtomicBool = AtomicBool::new(false);
static TERMINAL_RESIZED: AtomicBool = AtomicBool::new(false);

pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

pub fn terminated() -> bool {
    TERMINATED.load(Ordering::SeqCst)
}

/// Consume a pending terminal resize notification.
pub fn take_terminal_resize() -> bool {
    TERMINAL_RESIZED.swap(false, Ordering::SeqCst)
}

/// Test seam: raise the same flag the Unix signal and the Windows console
/// watcher raise, without touching OS handlers.
pub fn notify_terminal_resize() {
    TERMINAL_RESIZED.store(true, Ordering::SeqCst);
}

/// Enable ANSI escape processing for the run, restoring the previous console
/// mode when dropped. No-op on Unix; on Windows turns on
/// VIRTUAL_TERMINAL_PROCESSING for stdout (best effort: a redirected handle
/// or an old console just keeps working without colors) and puts the saved
/// mode back so the user's shell is left as it was found.
pub struct ConsoleModeGuard {
    _priv: (),
}

impl ConsoleModeGuard {
    pub fn enable() -> Self {
        #[cfg(windows)]
        windows::enable_virtual_terminal();
        // `panic = "abort"` in release skips `Drop`, but panic hooks still
        // run first — so the saved mode goes back even on that path.
        #[cfg(windows)]
        windows::install_panic_hook();
        ConsoleModeGuard { _priv: () }
    }
}

impl Drop for ConsoleModeGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        windows::restore_console_mode();
    }
}

#[cfg(unix)]
pub use unix::{
    die_from_sigterm, install_sigint_handler, install_sigterm_handler, install_sigwinch_handler,
    restore_sigpipe,
};

#[cfg(windows)]
pub use windows::{
    die_from_sigterm, install_sigint_handler, install_sigterm_handler, install_sigwinch_handler,
    restore_sigpipe,
};

#[cfg(unix)]
mod unix {
    use super::{INTERRUPTED, TERMINATED, TERMINAL_RESIZED};
    use std::sync::atomic::Ordering;

    /// SIGINT is recorded and checked from the run loop so teardown (cursor
    /// restore) happens through normal control flow — Drop alone would not run on
    /// a raw signal exit (plan.md §8).
    pub fn install_sigint_handler() {
        // SAFETY: signal(2) with a signal-safe handler that only stores a flag.
        unsafe {
            libc_signal(SIGINT, handle_sigint as *const () as usize);
        }
    }

    extern "C" fn handle_sigint(_: i32) {
        INTERRUPTED.store(true, Ordering::SeqCst);
    }

    /// SIGTERM is recorded like SIGINT so a supervisor killing an animation gets
    /// the normal teardown instead of a hidden cursor. `die_from_sigterm` then
    /// finishes the job the handler deferred.
    pub fn install_sigterm_handler() {
        // SAFETY: signal(2) with a signal-safe handler that only stores a flag.
        unsafe {
            libc_signal(SIGTERM, handle_sigterm as *const () as usize);
        }
    }

    extern "C" fn handle_sigterm(_: i32) {
        TERMINATED.store(true, Ordering::SeqCst);
    }

    /// Finish the SIGTERM we deferred: the cursor is back, so hand the signal to
    /// the default action and die from it. A supervisor then sees a terminated
    /// child, exactly as it would from the redirected run that never installs a
    /// handler at all. SIGINT does not go through here — upstream exits 1 on
    /// KeyboardInterrupt and parity outranks the convention (plan.md §8).
    pub fn die_from_sigterm() -> ! {
        // SAFETY: restoring the default action and re-raising is the documented
        // way to exit with a signal's status; raise(2) here does not return.
        unsafe {
            libc_signal(SIGTERM, SIG_DFL);
            libc_raise(SIGTERM);
        }
        unreachable!("SIGTERM with the default action terminates the process");
    }

    /// Record terminal resizes so the CLI can rebuild effects whose canvas and
    /// character positions were derived from the previous dimensions.
    pub fn install_sigwinch_handler() {
        // SAFETY: signal(2) with a signal-safe handler that only stores a flag.
        unsafe {
            libc_signal(SIGWINCH, handle_sigwinch as *const () as usize);
        }
    }

    extern "C" fn handle_sigwinch(_: i32) {
        TERMINAL_RESIZED.store(true, Ordering::SeqCst);
    }

    /// Restore default SIGPIPE so `ttfx ... | head` dies quietly like any Unix
    /// tool instead of panicking on a broken pipe (Rust ignores SIGPIPE by default).
    pub fn restore_sigpipe() {
        unsafe {
            libc_signal(SIGPIPE, SIG_DFL);
        }
    }

    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    const SIGPIPE: i32 = 13;
    /// 28 on Linux and on the BSDs, macOS included.
    const SIGWINCH: i32 = 28;
    const SIG_DFL: usize = 0;

    unsafe fn libc_signal(signum: i32, handler: usize) {
        unsafe extern "C" {
            fn signal(signum: i32, handler: usize) -> usize;
        }
        unsafe {
            signal(signum, handler);
        }
    }

    unsafe fn libc_raise(signum: i32) {
        unsafe extern "C" {
            fn raise(signum: i32) -> i32;
        }
        unsafe {
            raise(signum);
        }
    }
}

/// Windows platform: no new crates, raw kernel32 via `extern "system"`.
/// Ctrl-C and resize feed the same Atomics the Unix signals feed, so the run
/// loop, teardown, and `resize_settled()` debounce stay shared.
#[cfg(windows)]
mod windows {
    use super::{INTERRUPTED, TERMINAL_RESIZED};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    type Handle = isize;
    type Dword = u32;
    type Bool = i32;

    const STD_OUTPUT_HANDLE: Dword = 0xFFFF_FFF5;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: Dword = 0x0004;
    const CTRL_C_EVENT: Dword = 0;
    const CTRL_BREAK_EVENT: Dword = 1;

    static SAVED_OUTPUT_MODE: AtomicU32 = AtomicU32::new(0);
    static OUTPUT_MODE_SAVED: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" {
        fn GetStdHandle(nStdHandle: Dword) -> Handle;
        fn GetConsoleMode(hConsoleHandle: Handle, lpMode: *mut Dword) -> Bool;
        fn SetConsoleMode(hConsoleHandle: Handle, dwMode: Dword) -> Bool;
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(Dword) -> Bool>, add: Bool)
        -> Bool;
    }

    pub(super) fn enable_virtual_terminal() {
        // SAFETY: plain kernel32 queries on the output handle; failure
        // (redirected handle, old console) just means colors stay off.
        // The previous mode is saved so the Drop guard can put it back.
        unsafe {
            let out = GetStdHandle(STD_OUTPUT_HANDLE);
            if out == 0 || out == -1 {
                return;
            }
            let mut mode: Dword = 0;
            if GetConsoleMode(out, &mut mode) == 0 {
                return;
            }
            SAVED_OUTPUT_MODE.store(mode, Ordering::SeqCst);
            OUTPUT_MODE_SAVED.store(true, Ordering::SeqCst);
            let _ = SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }

    /// Restore the saved output mode from a panic hook. Hooks run before
    /// `abort`, unlike `Drop`; chaining keeps any previously installed hook
    /// (test harness, supervisor) alive. Re-entry safe via the same swap flag
    /// the `Drop` path uses.
    pub(super) fn install_panic_hook() {
        use std::sync::Once;
        static INSTALL: Once = Once::new();
        INSTALL.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                restore_console_mode();
                previous(info);
            }));
        });
    }

    pub(super) fn restore_console_mode() {
        if !OUTPUT_MODE_SAVED.swap(false, Ordering::SeqCst) {
            return;
        }
        let saved = SAVED_OUTPUT_MODE.load(Ordering::SeqCst);
        // SAFETY: restoring a mode previously read from the same handle.
        unsafe {
            let out = GetStdHandle(STD_OUTPUT_HANDLE);
            if out == 0 || out == -1 {
                return;
            }
            let _ = SetConsoleMode(out, saved);
        }
    }

    /// Ctrl-C / Ctrl-Break sets INTERRUPTED so the run loop tears down the
    /// cursor through normal control flow, exactly like SIGINT on Unix.
    pub fn install_sigint_handler() {
        // SAFETY: handler only stores an atomic flag. If registration itself
        // fails there is nothing safe to do here: Ctrl-C then terminates the
        // process with default handling, skipping the cursor teardown.
        unsafe {
            if SetConsoleCtrlHandler(Some(handle_ctrl), 1) == 0 {
                crate::errln!("Warning: failed to install console Ctrl handler.");
            }
        }
    }

    unsafe extern "system" fn handle_ctrl(ctrl_type: Dword) -> Bool {
        if ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT {
            INTERRUPTED.store(true, Ordering::SeqCst);
            1
        } else {
            0
        }
    }

    /// No SIGTERM on Windows; kept so `main.rs` stays platform-free.
    pub fn install_sigterm_handler() {}

    /// Windows never delivers SIGTERM to console apps; exit 1 after teardown.
    /// `exit` skips destructors, so restore the console mode explicitly here
    /// (the Drop guard cannot run on this path).
    pub fn die_from_sigterm() -> ! {
        restore_console_mode();
        std::process::exit(1);
    }

    /// No SIGPIPE on Windows.
    pub fn restore_sigpipe() {}

    /// Poll the terminal size on a parked thread and set the same
    /// TERMINAL_RESIZED flag SIGWINCH sets on Unix. An event-driven watcher
    /// would need the console input buffer (`ENABLE_WINDOW_INPUT`) and would
    /// consume every record it reads — including keystrokes meant for the
    /// shell. The sample period stays below `resize_settled()`'s quiet window
    /// so a sustained drag keeps pushing the timestamp forward and coalesces
    /// into one restart, exactly like a burst of SIGWINCH on Unix. Works when
    /// stdin is a pipe (the normal `echo hi | ttfx` case).
    ///
    /// The baseline is read on the calling thread: a resize landing between
    /// the engine's own first sample and the watcher's would otherwise be
    /// recorded as "no change" and missed until the next resize.
    pub fn install_sigwinch_handler() {
        let mut last = current_size();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(30));
                let now = current_size();
                if now != last {
                    last = now;
                    TERMINAL_RESIZED.store(true, Ordering::SeqCst);
                }
            }
        });
    }

    fn current_size() -> Option<(u16, u16)> {
        terminal_size::terminal_size().map(|(w, h)| (w.0, h.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_resize_notifications_are_consumed() {
        take_terminal_resize();
        notify_terminal_resize();
        assert!(take_terminal_resize());
        assert!(!take_terminal_resize());
    }
}
