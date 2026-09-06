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

/// Enable ANSI escape processing. No-op on Unix; on Windows turns on
/// VIRTUAL_TERMINAL_PROCESSING for stdout (best effort: a redirected handle
/// or an old console just keeps working without colors).
pub fn enable_ansi() {
    #[cfg(windows)]
    windows::enable_virtual_terminal();
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
    use std::sync::atomic::Ordering;

    type Handle = isize;
    type Dword = u32;
    type Bool = i32;

    const STD_OUTPUT_HANDLE: Dword = 0xFFFF_FFF5;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: Dword = 0x0004;
    const WINDOW_BUFFER_SIZE_EVENT: u16 = 0x0004;
    const CTRL_C_EVENT: Dword = 0;
    const CTRL_BREAK_EVENT: Dword = 1;

    unsafe extern "system" {
        fn GetStdHandle(nStdHandle: Dword) -> Handle;
        fn GetConsoleMode(hConsoleHandle: Handle, lpMode: *mut Dword) -> Bool;
        fn SetConsoleMode(hConsoleHandle: Handle, dwMode: Dword) -> Bool;
        fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(Dword) -> Bool>, add: Bool)
        -> Bool;
        fn CreateFileW(
            lpFileName: *const u16,
            dwDesiredAccess: Dword,
            dwShareMode: Dword,
            lpSecurityAttributes: *const u8,
            dwCreationDisposition: Dword,
            dwFlagsAndAttributes: Dword,
            hTemplateFile: Handle,
        ) -> Handle;
        fn ReadConsoleInputW(
            hConsoleInput: Handle,
            lpBuffer: *mut InputRecord,
            nLength: Dword,
            lpNumberOfEventsRead: *mut Dword,
        ) -> Bool;
        fn CloseHandle(hObject: Handle) -> Bool;
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct InputRecord {
        event_type: u16,
        reserved: u16,
        payload: [u8; 16],
    }

    pub(super) fn enable_virtual_terminal() {
        // SAFETY: plain kernel32 queries on the output handle; failure
        // (redirected handle, old console) just means colors stay off.
        unsafe {
            let out = GetStdHandle(STD_OUTPUT_HANDLE);
            if out == 0 || out == -1 {
                return;
            }
            let mut mode: Dword = 0;
            if GetConsoleMode(out, &mut mode) == 0 {
                return;
            }
            let _ = SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }

    /// Ctrl-C / Ctrl-Break sets INTERRUPTED so the run loop tears down the
    /// cursor through normal control flow, exactly like SIGINT on Unix.
    pub fn install_sigint_handler() {
        // SAFETY: handler only stores an atomic flag.
        unsafe {
            SetConsoleCtrlHandler(Some(handle_ctrl), 1);
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
    pub fn die_from_sigterm() -> ! {
        std::process::exit(1);
    }

    /// No SIGPIPE on Windows.
    pub fn restore_sigpipe() {}

    /// Watch `CONIN$` for WINDOW_BUFFER_SIZE_EVENTs on a parked thread and set
    /// the same TERMINAL_RESIZED flag SIGWINCH sets on Unix. `CONIN$` is opened
    /// explicitly so this works even when stdin itself is a pipe
    /// (`echo hi | ttfx`), which is the normal case.
    pub fn install_sigwinch_handler() {
        std::thread::spawn(|| {
            // SAFETY: minimal kernel32 console input pump; any failure ends
            // the thread quietly — resize-restart just stays off.
            unsafe { watch_for_resize() };
        });
    }

    unsafe fn watch_for_resize() {
        const GENERIC_READ: Dword = 0x8000_0000;
        const GENERIC_WRITE: Dword = 0x4000_0000;
        const FILE_SHARE_READ: Dword = 1;
        const FILE_SHARE_WRITE: Dword = 2;
        const OPEN_EXISTING: Dword = 3;

        // "CONIN$" as UTF-16 + NUL.
        let name: [u16; 7] = [0x43, 0x4F, 0x4E, 0x49, 0x4E, 0x24, 0];
        let conin = CreateFileW(
            name.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_EXISTING,
            0,
            0,
        );
        if conin == 0 || conin == -1 {
            return;
        }
        let mut record = InputRecord { event_type: 0, reserved: 0, payload: [0; 16] };
        loop {
            let mut read: Dword = 0;
            let ok = ReadConsoleInputW(conin, &mut record, 1, &mut read);
            if ok == 0 || read == 0 {
                break;
            }
            if record.event_type == WINDOW_BUFFER_SIZE_EVENT {
                TERMINAL_RESIZED.store(true, Ordering::SeqCst);
            }
        }
        CloseHandle(conin);
    }

    #[allow(dead_code)]
    pub(super) fn set_resized_for_test() {
        TERMINAL_RESIZED.store(true, Ordering::SeqCst);
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
