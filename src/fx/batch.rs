//! Batched motion steps.
//!
//! update ticks the active set a bitmap word (64 slots) at a time. Before it
//! ticks a word, `motion_batch` works out the next step of every character
//! in it whose mirror (motion.rs) describes its active path, 8 lanes wide
//! with AVX-512 or 4 with AVX2, instead of one path walk per tick. For
//! parity each lane does `mirror_step`'s divide, multiplies and adds, with
//! no FMA, and cvtpd2dq's half-to-even rounding (a lane outside i32 converts
//! to i32::MIN and is left to the scalar step). Steps and distances go to the
//! mirrors at once (the old distances kept in `Batch`); coordinates wait in
//! `Batch` until update ticks the character (`motion_apply`).
//!
//! A step depends only on the character's own mirror, which only its own
//! tick changes - or an effect callback, or a change to a path another
//! character shares (which drops the mirror). So a result stands while the
//! mirror still describes the active path and no callback ran since the
//! batch (`motion_epoch`). Only a callback reads or changes another
//! character's path, so before one runs, the steps of the characters not
//! ticked yet are taken back (`motion_settle`). (A tick can also drop
//! another character's mirror when both are active on one path; no effect
//! shares an active path, and mirrors already assume none does.)
//!
//! Idle ticks. A step that neither moves the character nor ends the path is
//! unobservable, and so is step_animation for a character without a scene,
//! with no frames left, or with a synced, non-looping scene whose frame at
//! the new step is the one it shows. Those lanes come back in `idle`, and
//! update leaves them out of the word (a callback puts back the ones not
//! reached yet, `motion_settle`). The vector pass works out that frame index
//! with each step, for the sync key the slot last asked with
//! (`Mirrors::sync`); a batched tick that does more takes it too
//! (`take_sync_index`), sparing step_synced_scene its divides and the path
//! record.
//!
//! Below x86-64-v3 there is no batch: every step goes through the scalar
//! `mirror_step` or the walk.

use crate::utils::geometry::Coord;
use crate::utils::pycompat::round_half_even;

use super::motion::Mirrors;
use super::{At, Engine, Hooks, NONE};

/// A frame index motion_batch could not work out (cvtpd2dq's out of range).
pub(super) const NO_INDEX: u32 = 1 << 31;
/// A sync key's bit for a step-synced scene.
pub(super) const SYNC_KEY_STEP: u32 = 1 << 31;

/// A word's worked-out steps, by slot & 63.
pub struct Batch {
    step: [f64; 64],
    d: [f64; 64],
    /// The mirrors' distances before the batch (`restore`).
    old: [f64; 64],
    x: [i32; 64],
    y: [i32; 64],
    /// The synced scene's frame index at the step, for the slot's sync key
    /// (valid while `Mirrors::synced`; NO_INDEX when unknown).
    sidx: [u32; 64],
    /// The lanes whose tick does nothing but their step.
    pub(super) idle: u64,
    /// Quiet lanes with a scene: idle when the scene does nothing.
    quiet: u64,
    /// The lanes whose tick only moves a character without a scene.
    pub(super) bare: u64,
    /// The lanes of the word being ticked whose steps are in the mirrors.
    pub(super) live: u64,
    /// The slot whose tick takes its synced frame index from `sidx`, NONE.
    pub(super) hint: u32,
    /// The kernel the CPU runs: 0 none, 3 AVX2, 4 AVX-512.
    tier: u8,
}

impl Default for Batch {
    fn default() -> Self {
        Batch {
            step: [0.0; 64],
            d: [0.0; 64],
            old: [0.0; 64],
            x: [0; 64],
            y: [0; 64],
            sidx: [NO_INDEX; 64],
            idle: 0,
            quiet: 0,
            bare: 0,
            live: 0,
            hint: NONE,
            tier: tier(),
        }
    }
}

/// step_synced_scene's frame index for `key` at (step, d) of a path with
/// (max_steps, total_distance), or NO_INDEX outside i32 (as cvtpd2dq).
#[inline(always)]
fn sync_index(key: u32, step: f64, max: f64, total: f64, d: f64) -> u32 {
    let last = (key & !SYNC_KEY_STEP) as i64 - 1;
    let ratio = if key & SYNC_KEY_STEP != 0 {
        step.max(1.0) / max.max(1.0)
    } else {
        let whole = total.max(1.0);
        let remaining = (total - d).max(1.0);
        (whole - remaining).max(1.0) / whole
    };
    let index = round_half_even(last as f64 * ratio);
    if index <= i32::MIN as i64 || index > i32::MAX as i64 {
        return NO_INDEX;
    }
    index.min(last).max(0) as u32
}

/// The widest kernel the CPU runs. TTFX_NO_AVX512 and TTFX_NO_AVX2 (for
/// testing) leave out the kernels that need them.
fn tier() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        let off = |name: &str| std::env::var_os(name).is_some();
        if off("TTFX_NO_AVX2") {
            return 0;
        }
        if !off("TTFX_NO_AVX512") && has!("avx512f") && has!("avx512vl") && has!("avx2") {
            return 4;
        }
        if has!("avx2") {
            return 3;
        }
    }
    0
}

impl Batch {
    /// The scalar step of lane `k` (slot base + k); its bit when done.
    #[inline(always)]
    fn one(&mut self, m: &mut Mirrors, etab: &[f64], base: usize, k: usize) -> u64 {
        let slot = (base + k) as u32;
        if let Some(r) = m.next_step(etab, slot) {
            let (x, y) = (round_half_even(r.x), round_half_even(r.y));
            if (i32::MIN as i64) < x.min(y) && x.max(y) <= i32::MAX as i64 {
                self.step[k] = r.step;
                self.d[k] = r.d;
                self.old[k] = *m.last.at(slot);
                *m.step.at_mut(slot) = r.step;
                *m.last.at_mut(slot) = r.d;
                self.x[k] = x as i32;
                self.y[k] = y as i32;
                return 1 << k;
            }
        }
        0
    }

    /// Classify the step of lane `k` (done by `one`) as the kernels do.
    #[inline(always)]
    fn classify(&mut self, m: &Mirrors, coord: &[Coord], scene: &[u32], base: usize, k: usize) {
        let slot = (base + k) as u32;
        if m.synced {
            let (max, total) = (*m.max.at(slot), *m.total.at(slot));
            self.sidx[k] = sync_index(*m.sync.at(slot), self.step[k], max, total, self.d[k]);
        }
        if self.step[k] == *m.max.at(slot) {
            return;
        }
        let still = *coord.at(slot) == Coord::new(self.x[k] as i64, self.y[k] as i64);
        match (still, *scene.at(slot) == NONE) {
            (true, true) => self.idle |= 1 << k,
            (true, false) => self.quiet |= 1 << k,
            (false, true) => self.bare |= 1 << k,
            (false, false) => {}
        }
    }

    /// Take word `w`'s steps in `lanes` back out of their mirrors.
    pub(super) fn restore(&self, m: &mut Mirrors, w: usize, mut lanes: u64) {
        while lanes != 0 {
            let k = lanes.trailing_zeros() as usize;
            lanes &= lanes - 1;
            let slot = (w * 64 + k) as u32;
            // whole numbers below 2^53: exact
            *m.step.at_mut(slot) = self.step[k] - 1.0;
            *m.last.at_mut(slot) = self.old[k];
        }
    }

    /// The synced frame index motion_batch worked out for this tick of
    /// `slot`, if it is the tick the batch was for; NO_INDEX otherwise.
    #[inline(always)]
    pub(super) fn take_sync_index(&mut self, slot: u32) -> u32 {
        if self.hint != slot {
            return NO_INDEX;
        }
        self.hint = NONE;
        self.sidx[(slot & 63) as usize]
    }
}

impl Engine {
    /// Work out the steps of the word `w`'s characters in `snapshot` whose
    /// mirror describes their active path; returns the bits of the ones
    /// done (0 without AVX2).
    #[inline]
    pub(super) fn motion_batch(&mut self, w: usize, snapshot: u64) -> u64 {
        self.batch.idle = 0;
        self.batch.quiet = 0;
        self.batch.bare = 0;
        let done = match self.batch.tier {
            // SAFETY: the CPU supports these (checked when Batch was made).
            #[cfg(target_arch = "x86_64")]
            4 => unsafe { self.motion_batch_avx512(w, snapshot) },
            // SAFETY: as above.
            #[cfg(target_arch = "x86_64")]
            3 => unsafe { self.motion_batch_avx2(w, snapshot) },
            _ => return 0,
        };
        let mut quiet = self.batch.quiet;
        while quiet != 0 {
            let k = quiet.trailing_zeros() as usize;
            quiet &= quiet - 1;
            let slot = (w * 64 + k) as u32;
            if self.animation_idle(slot, *self.ch.scene.at(slot), self.batch.sidx[k]) {
                self.batch.idle |= 1 << k;
            }
        }
        done
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn motion_batch_avx2(&mut self, w: usize, snapshot: u64) -> u64 {
        use std::arch::x86_64::*;
        use super::motion::{MF_CLAMP, MF_CURVE, MF_LOWER, MF_OVER};

        let (m, etab) = self.paths.mirrors_etab();
        let (ch_path, ch_coord, ch_scene) = (&self.ch.path[..], &self.ch.coord[..], &self.ch.scene[..]);
        assert!(ch_coord.len() == ch_path.len() && ch_scene.len() == ch_path.len());
        let b = &mut self.batch;
        let mut done = 0u64;
        let base = w * 64;
        // SAFETY: every group read lies inside the mirror arrays (sized to
        // whole groups of 8 past every slot) and inside ch.path, ch.coord
        // and ch.scene (checked; Coord is repr(C), two i64s);
        // the gather reads etab only at lanes whose mirror names a table
        // offset whose entries run past the step (step <= max_steps).
        unsafe {
            let one = _mm256_set1_pd(1.0);
            let zero = _mm256_setzero_pd();
            let none = _mm_set1_epi32(NONE as i32);
            let int_min = _mm_set1_epi32(i32::MIN);
            let mut groups = snapshot;
            while groups != 0 {
                let g = groups.trailing_zeros() as usize / 4;
                let lanes = (groups >> (g * 4)) as u32 & 0xf;
                groups &= !(0xfu64 << (g * 4));
                let s = base + g * 4;
                if s + 4 > ch_path.len() {
                    continue;
                }
                debug_assert!(s + 4 <= m.path.len());
                let mp = _mm_loadu_si128(m.path.as_ptr().add(s) as *const __m128i);
                let cp = _mm_loadu_si128(ch_path.as_ptr().add(s) as *const __m128i);
                let mirrored = _mm_andnot_si128(_mm_cmpeq_epi32(mp, none), _mm_cmpeq_epi32(mp, cp));
                let lanes = lanes & _mm_movemask_ps(_mm_castsi128_ps(mirrored)) as u32;
                if lanes == 0 {
                    continue;
                }
                if lanes & (lanes - 1) == 0 {
                    // one lane: the scalar step is cheaper
                    let k = g * 4 + lanes.trailing_zeros() as usize;
                    let bit = b.one(m, etab, base, k);
                    if bit != 0 {
                        b.classify(m, ch_coord, ch_scene, base, k);
                    }
                    done |= bit;
                    continue;
                }
                let step = _mm256_add_pd(_mm256_loadu_pd(m.step.as_ptr().add(s)), one);
                let max = _mm256_loadu_pd(m.max.as_ptr().add(s));
                let valid = _mm256_cmp_pd::<_CMP_LE_OQ>(step, max);
                let tab = _mm_loadu_si128(m.etab.as_ptr().add(s) as *const __m128i);
                let linear = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(_mm_cmpeq_epi32(tab, _mm_setzero_si128())));
                let bit = _mm256_setr_epi64x(1, 2, 4, 8);
                let live = _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x(lanes as i64), bit), bit));
                let gather = _mm256_and_pd(_mm256_andnot_pd(linear, valid), live);
                let ratio = _mm256_div_pd(step, max);
                let factor = if _mm256_movemask_pd(gather) == 0 {
                    ratio
                } else {
                    let index = _mm_add_epi32(tab, _mm256_cvttpd_epi32(step));
                    let eased = _mm256_mask_i32gather_pd::<8>(zero, etab.as_ptr(), index, gather);
                    _mm256_blendv_pd(eased, ratio, linear)
                };
                let total = _mm256_loadu_pd(m.total.as_ptr().add(s));
                let d = _mm256_mul_pd(factor, total);
                let off = _mm256_loadu_pd(m.off.as_ptr().add(s));
                let hi = _mm256_loadu_pd(m.hi.as_ptr().add(s));
                let flags = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(i32::from_ne_bytes(
                    std::ptr::read_unaligned(m.flags.as_ptr().add(s) as *const [u8; 4]),
                )));
                let flag = |bit: u8| {
                    let v = _mm256_set1_epi64x(bit as i64);
                    _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(flags, v), v))
                };
                let (lower, curve, clamp, over) = (flag(MF_LOWER), flag(MF_CURVE), flag(MF_CLAMP), flag(MF_OVER));
                let r = _mm256_sub_pd(d, off);
                let within = _mm256_or_pd(_mm256_cmp_pd::<_CMP_LE_OQ>(r, hi), over);
                let r = _mm256_blendv_pd(r, _mm256_add_pd(r, hi), over);
                let above = _mm256_or_pd(_mm256_andnot_pd(lower, _mm256_castsi256_pd(_mm256_set1_epi64x(-1))), _mm256_cmp_pd::<_CMP_GT_OQ>(d, off));
                let ok = _mm256_and_pd(_mm256_and_pd(valid, within), above);
                // t = r / hi (0 for an empty segment; a linear ratio at most
                // 1.0, and min picks 1.0 over NaN as f64::min does)
                let q = _mm256_div_pd(r, hi);
                let t = _mm256_blendv_pd(q, _mm256_min_pd(q, one), clamp);
                let t = _mm256_andnot_pd(_mm256_cmp_pd::<_CMP_EQ_OQ>(hi, zero), t);
                let u = _mm256_sub_pd(one, t);
                let lerp = |a: __m256d, b: __m256d| _mm256_add_pd(_mm256_mul_pd(u, a), _mm256_mul_pd(t, b));
                // a line's control is its end: one lerp unless a lane curves
                let curves = _mm256_movemask_pd(_mm256_and_pd(curve, live)) != 0;
                let point = |start: *const f64, control: *const f64, end: *const f64| {
                    let (a, e) = (_mm256_loadu_pd(start.add(s)), _mm256_loadu_pd(end.add(s)));
                    let line = lerp(a, e);
                    if !curves {
                        return _mm256_cvtpd_epi32(line);
                    }
                    let c = _mm256_loadu_pd(control.add(s));
                    let bent = lerp(lerp(a, c), lerp(c, e));
                    _mm256_cvtpd_epi32(_mm256_blendv_pd(line, bent, curve))
                };
                let x = point(m.sx.as_ptr(), m.cx.as_ptr(), m.ex.as_ptr());
                let y = point(m.sy.as_ptr(), m.cy.as_ptr(), m.ey.as_ptr());
                let wide = _mm_or_si128(_mm_cmpeq_epi32(x, int_min), _mm_cmpeq_epi32(y, int_min));
                let lanes = lanes
                    & _mm256_movemask_pd(ok) as u32
                    & !(_mm_movemask_ps(_mm_castsi128_ps(wide)) as u32);
                let k = g * 4;
                _mm256_storeu_pd(b.step.as_mut_ptr().add(k), step);
                _mm256_storeu_pd(b.d.as_mut_ptr().add(k), d);
                let taken = _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x(lanes as i64), bit), bit);
                _mm256_storeu_pd(b.old.as_mut_ptr().add(k), _mm256_loadu_pd(m.last.as_ptr().add(s)));
                _mm256_maskstore_pd(m.step.as_mut_ptr().add(s), taken, step);
                _mm256_maskstore_pd(m.last.as_mut_ptr().add(s), taken, d);
                _mm_storeu_si128(b.x.as_mut_ptr().add(k) as *mut __m128i, x);
                _mm_storeu_si128(b.y.as_mut_ptr().add(k) as *mut __m128i, y);
                done |= (lanes as u64) << k;
                let pairs = ch_coord.as_ptr().add(s) as *const __m256i;
                let (c0, c1) = (_mm256_loadu_si256(pairs), _mm256_loadu_si256(pairs.add(1)));
                let cols = _mm256_permute4x64_epi64::<0xd8>(_mm256_unpacklo_epi64(c0, c1));
                let rows = _mm256_permute4x64_epi64::<0xd8>(_mm256_unpackhi_epi64(c0, c1));
                let same = _mm256_and_si256(
                    _mm256_cmpeq_epi64(cols, _mm256_cvtepi32_epi64(x)),
                    _mm256_cmpeq_epi64(rows, _mm256_cvtepi32_epi64(y)),
                );
                let same = _mm256_movemask_pd(_mm256_castsi256_pd(same)) as u32;
                let on = lanes & _mm256_movemask_pd(_mm256_cmp_pd::<_CMP_NEQ_UQ>(step, max)) as u32;
                let scene = _mm_loadu_si128(ch_scene.as_ptr().add(s) as *const __m128i);
                let bare = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(scene, none))) as u32;
                b.idle |= ((on & same & bare) as u64) << k;
                b.quiet |= ((on & same & !bare) as u64) << k;
                b.bare |= ((on & !same & bare) as u64) << k;
                if m.synced {
                    // step_synced_scene's frame index at the new step
                    let key = _mm_loadu_si128(m.sync.as_ptr().add(s) as *const __m128i);
                    let by_step = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(key));
                    let ratio_step = _mm256_div_pd(_mm256_max_pd(step, one), _mm256_max_pd(max, one));
                    let whole = _mm256_max_pd(total, one);
                    let remaining = _mm256_max_pd(_mm256_sub_pd(total, d), one);
                    let ratio_d = _mm256_div_pd(_mm256_max_pd(_mm256_sub_pd(whole, remaining), one), whole);
                    let ratio = _mm256_blendv_pd(ratio_d, ratio_step, by_step);
                    let last = _mm_sub_epi32(_mm_and_si128(key, _mm_set1_epi32(i32::MAX)), _mm_set1_epi32(1));
                    let index = _mm256_cvtpd_epi32(_mm256_mul_pd(_mm256_cvtepi32_pd(last), ratio));
                    let unknown = _mm_cmpeq_epi32(index, int_min);
                    let index = _mm_max_epi32(_mm_min_epi32(index, last), _mm_setzero_si128());
                    let index = _mm_blendv_epi8(index, int_min, unknown);
                    _mm_storeu_si128(b.sidx.as_mut_ptr().add(k) as *mut __m128i, index);
                }
            }
        }
        done
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vl,avx2")]
    unsafe fn motion_batch_avx512(&mut self, w: usize, snapshot: u64) -> u64 {
        use std::arch::x86_64::*;
        use super::motion::{MF_CLAMP, MF_CURVE, MF_LOWER, MF_OVER};

        let (m, etab) = self.paths.mirrors_etab();
        let (ch_path, ch_coord, ch_scene) = (&self.ch.path[..], &self.ch.coord[..], &self.ch.scene[..]);
        assert!(ch_coord.len() == ch_path.len() && ch_scene.len() == ch_path.len());
        let b = &mut self.batch;
        let mut done = 0u64;
        let base = w * 64;
        // SAFETY: as for motion_batch_avx2, with groups of 8 (the mirror
        // arrays are whole words long).
        unsafe {
            let one = _mm512_set1_pd(1.0);
            let zero = _mm512_setzero_pd();
            let none = _mm256_set1_epi32(NONE as i32);
            let int_min = _mm256_set1_epi32(i32::MIN);
            let mut groups = snapshot;
            while groups != 0 {
                let g = groups.trailing_zeros() as usize / 8;
                let lanes = (groups >> (g * 8)) as u8;
                groups &= !(0xffu64 << (g * 8));
                let s = base + g * 8;
                if s + 8 > ch_path.len() {
                    let mut l = lanes;
                    while l != 0 {
                        let k = g * 8 + l.trailing_zeros() as usize;
                        l &= l - 1;
                        if base + k < ch_path.len() && m.path[base + k] != NONE && m.path[base + k] == ch_path[base + k] {
                            let bit = b.one(m, etab, base, k);
                            if bit != 0 {
                                b.classify(m, ch_coord, ch_scene, base, k);
                            }
                            done |= bit;
                        }
                    }
                    continue;
                }
                debug_assert!(s + 8 <= m.path.len());
                let mp = _mm256_loadu_si256(m.path.as_ptr().add(s) as *const __m256i);
                let cp = _mm256_loadu_si256(ch_path.as_ptr().add(s) as *const __m256i);
                let lanes = lanes & _mm256_cmpeq_epi32_mask(mp, cp) & _mm256_cmpneq_epi32_mask(mp, none);
                if lanes == 0 {
                    continue;
                }
                if lanes & (lanes - 1) == 0 {
                    let k = g * 8 + lanes.trailing_zeros() as usize;
                    let bit = b.one(m, etab, base, k);
                    if bit != 0 {
                        b.classify(m, ch_coord, ch_scene, base, k);
                    }
                    done |= bit;
                    continue;
                }
                let step = _mm512_add_pd(_mm512_loadu_pd(m.step.as_ptr().add(s)), one);
                let max = _mm512_loadu_pd(m.max.as_ptr().add(s));
                let valid = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(step, max);
                let tab = _mm256_loadu_si256(m.etab.as_ptr().add(s) as *const __m256i);
                let linear = _mm256_cmpeq_epi32_mask(tab, _mm256_setzero_si256());
                let ratio = _mm512_div_pd(step, max);
                let gather = !linear & valid & lanes;
                let factor = if gather == 0 {
                    ratio
                } else {
                    let index = _mm256_add_epi32(tab, _mm512_cvttpd_epi32(step));
                    let eased = _mm512_mask_i32gather_pd::<8>(zero, gather, index, etab.as_ptr());
                    _mm512_mask_blend_pd(linear, eased, ratio)
                };
                let total = _mm512_loadu_pd(m.total.as_ptr().add(s));
                let d = _mm512_mul_pd(factor, total);
                let off = _mm512_loadu_pd(m.off.as_ptr().add(s));
                let hi = _mm512_loadu_pd(m.hi.as_ptr().add(s));
                let flags = _mm512_cvtepu8_epi64(_mm_loadl_epi64(m.flags.as_ptr().add(s) as *const __m128i));
                let flag = |bit: u8| _mm512_test_epi64_mask(flags, _mm512_set1_epi64(bit as i64));
                let (lower, curve, clamp, over) = (flag(MF_LOWER), flag(MF_CURVE), flag(MF_CLAMP), flag(MF_OVER));
                let r = _mm512_sub_pd(d, off);
                let within = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(r, hi) | over;
                let r = _mm512_mask_add_pd(r, over, r, hi);
                let above = !lower | _mm512_cmp_pd_mask::<_CMP_GT_OQ>(d, off);
                let ok = valid & within & above;
                // t = r / hi (0 for an empty segment; a linear ratio at most
                // 1.0, and min picks 1.0 over NaN as f64::min does)
                let q = _mm512_div_pd(r, hi);
                let t = _mm512_mask_blend_pd(clamp, q, _mm512_min_pd(q, one));
                let t = _mm512_mask_blend_pd(_mm512_cmp_pd_mask::<_CMP_EQ_OQ>(hi, zero), t, zero);
                let u = _mm512_sub_pd(one, t);
                let lerp = |a: __m512d, b: __m512d| _mm512_add_pd(_mm512_mul_pd(u, a), _mm512_mul_pd(t, b));
                let curves = curve & lanes != 0;
                let point = |start: *const f64, control: *const f64, end: *const f64| {
                    let (a, e) = (_mm512_loadu_pd(start.add(s)), _mm512_loadu_pd(end.add(s)));
                    let line = lerp(a, e);
                    if !curves {
                        return _mm512_cvtpd_epi32(line);
                    }
                    let c = _mm512_loadu_pd(control.add(s));
                    let bent = lerp(lerp(a, c), lerp(c, e));
                    _mm512_cvtpd_epi32(_mm512_mask_blend_pd(curve, line, bent))
                };
                let x = point(m.sx.as_ptr(), m.cx.as_ptr(), m.ex.as_ptr());
                let y = point(m.sy.as_ptr(), m.cy.as_ptr(), m.ey.as_ptr());
                let wide = _mm256_cmpeq_epi32_mask(x, int_min) | _mm256_cmpeq_epi32_mask(y, int_min);
                let lanes = lanes & ok & !wide;
                let k = g * 8;
                _mm512_storeu_pd(b.step.as_mut_ptr().add(k), step);
                _mm512_storeu_pd(b.d.as_mut_ptr().add(k), d);
                _mm512_storeu_pd(b.old.as_mut_ptr().add(k), _mm512_loadu_pd(m.last.as_ptr().add(s)));
                _mm512_mask_storeu_pd(m.step.as_mut_ptr().add(s), lanes, step);
                _mm512_mask_storeu_pd(m.last.as_mut_ptr().add(s), lanes, d);
                _mm256_storeu_si256(b.x.as_mut_ptr().add(k) as *mut __m256i, x);
                _mm256_storeu_si256(b.y.as_mut_ptr().add(k) as *mut __m256i, y);
                done |= (lanes as u64) << k;
                let pairs = ch_coord.as_ptr().add(s) as *const i64;
                let (c0, c1) = (_mm512_loadu_epi64(pairs), _mm512_loadu_epi64(pairs.add(8)));
                let cols = _mm512_permutex2var_epi64(c0, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), c1);
                let rows = _mm512_permutex2var_epi64(c0, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), c1);
                let same = _mm512_cmpeq_epi64_mask(cols, _mm512_cvtepi32_epi64(x))
                    & _mm512_cmpeq_epi64_mask(rows, _mm512_cvtepi32_epi64(y));
                let on = lanes & _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(step, max);
                let scene = _mm256_loadu_si256(ch_scene.as_ptr().add(s) as *const __m256i);
                let bare = _mm256_cmpeq_epi32_mask(scene, none);
                b.idle |= ((on & same & bare) as u64) << k;
                b.quiet |= ((on & same & !bare) as u64) << k;
                b.bare |= ((on & !same & bare) as u64) << k;
                if m.synced {
                    let key = _mm256_loadu_si256(m.sync.as_ptr().add(s) as *const __m256i);
                    let by_step = _mm256_cmplt_epi32_mask(key, _mm256_setzero_si256());
                    let ratio_step = _mm512_div_pd(_mm512_max_pd(step, one), _mm512_max_pd(max, one));
                    let whole = _mm512_max_pd(total, one);
                    let remaining = _mm512_max_pd(_mm512_sub_pd(total, d), one);
                    let ratio_d = _mm512_div_pd(_mm512_max_pd(_mm512_sub_pd(whole, remaining), one), whole);
                    let ratio = _mm512_mask_blend_pd(by_step, ratio_d, ratio_step);
                    let last = _mm256_sub_epi32(_mm256_and_si256(key, _mm256_set1_epi32(i32::MAX)), _mm256_set1_epi32(1));
                    let index = _mm512_cvtpd_epi32(_mm512_mul_pd(_mm512_cvtepi32_pd(last), ratio));
                    let unknown = _mm256_cmpeq_epi32_mask(index, int_min);
                    let index = _mm256_max_epi32(_mm256_min_epi32(index, last), _mm256_setzero_si256());
                    let index = _mm256_mask_mov_epi32(index, unknown, int_min);
                    _mm256_storeu_si256(b.sidx.as_mut_ptr().add(k) as *mut __m256i, index);
                }
            }
        }
        done
    }

    /// Motion.move for a character `motion_batch` stepped (the mirror has the
    /// step): the new coordinate, then the rest of Motion.move at max_steps.
    #[inline(always)]
    pub(super) fn motion_apply(&mut self, hooks: &mut dyn Hooks, slot: u32) {
        let lane = (slot & 63) as usize;
        let b = &self.batch;
        let (step, coord) = (b.step[lane], Coord::new(b.x[lane] as i64, b.y[lane] as i64));
        let reached = step == *self.paths.m.max.at(slot);
        if *self.ch.coord.at(slot) != coord {
            self.set_coordinate(slot, coord);
        }
        if reached {
            // the tail reads the record
            self.paths.release(slot);
            self.motion_tail(hooks, slot);
        } else {
            // the path goes on: this tick's synced frame index stands
            self.batch.hint = slot;
        }
    }

    /// A bare move (`Batch::bare`): the new coordinate (the mirror has the
    /// step), and nothing else happens on this tick.
    #[inline(always)]
    pub(super) fn bare_move(&mut self, slot: u32) {
        let lane = (slot & 63) as usize;
        let b = &self.batch;
        self.set_coordinate(slot, Coord::new(b.x[lane] as i64, b.y[lane] as i64));
    }

    /// The batch's result for this slot still stands: its mirror describes
    /// its active path and no callback ran since the batch.
    #[inline(always)]
    pub(super) fn batched(&self, slot: u32, epoch: u32) -> bool {
        let path = *self.paths.m.path.at(slot);
        self.motion_epoch == epoch && path != NONE && path == *self.ch.path.at(slot)
    }
}
