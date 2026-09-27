//! Unchecked slice access for the fixed-size and SIMD kernels.
//!
//! Every helper reads or writes the run of elements (or vector bytes) that
//! starts at index `i`, and that run must lie inside the slice. The kernels
//! index by construction (whole groups of lanes in arrays sized for them,
//! rows and pools kept with slack past their content; see fx::At), so only
//! debug builds check.

/// `a[i..i + N]` as an array.
#[inline(always)]
pub fn load<T: Copy, const N: usize>(a: &[T], i: usize) -> [T; N] {
    debug_assert!(i + N <= a.len(), "{N} at {i} out of bounds ({})", a.len());
    // SAFETY: in bounds by construction (see the module); unaligned.
    unsafe { std::ptr::read_unaligned(a.as_ptr().add(i) as *const [T; N]) }
}

/// `a[i..i + N] = v`.
#[inline(always)]
pub fn store<T: Copy, const N: usize>(a: &mut [T], i: usize, v: [T; N]) {
    debug_assert!(i + N <= a.len(), "{N} at {i} out of bounds ({})", a.len());
    // SAFETY: in bounds by construction (see the module); unaligned.
    unsafe { std::ptr::write_unaligned(a.as_mut_ptr().add(i) as *mut [T; N], v) }
}

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "x86_64")]
// calling one needs only its target feature, which the kernels have
#[allow(clippy::missing_safety_doc)]
mod x86 {
    use std::arch::x86_64::*;
    use std::mem::size_of;

    /// Element types an integer vector may load from and store to: no
    /// padding, and every bit pattern is a value.
    pub trait Plain: Copy + sealed::Sealed {}
    impl<T: Copy + sealed::Sealed> Plain for T {}

    mod sealed {
        pub trait Sealed {}
        impl Sealed for u8 {}
        impl Sealed for u32 {}
        impl Sealed for i32 {}
        impl Sealed for u64 {}
        impl Sealed for i64 {}
        // repr(C), two i64s
        impl Sealed for crate::utils::geometry::Coord {}
    }

    /// The `bytes` from element `i` lie inside `a`.
    #[inline(always)]
    fn fits<T>(a: &[T], i: usize, bytes: usize) -> bool {
        i * size_of::<T>() + bytes <= size_of_val(a)
    }

    /// Unaligned vector access to `a` from element `i`.
    macro_rules! access {
        ($feature:literal, $load:ident, $load2:ident, $store:ident, $vec:ty, [$($g:tt)*] $elem:ty, $ld:ident, $st:ident, $cast:ty) => {
            #[target_feature(enable = $feature)]
            #[inline]
            pub fn $load<$($g)*>(a: &[$elem], i: usize) -> $vec {
                debug_assert!(fits(a, i, size_of::<$vec>()));
                // SAFETY: in bounds by construction (see the module).
                unsafe { $ld(a.as_ptr().add(i) as *const $cast) }
            }

            /// Two vectors in a row.
            #[target_feature(enable = $feature)]
            #[inline]
            pub fn $load2<$($g)*>(a: &[$elem], i: usize) -> [$vec; 2] {
                debug_assert!(fits(a, i, 2 * size_of::<$vec>()));
                // SAFETY: in bounds by construction (see the module).
                unsafe {
                    let p = a.as_ptr().add(i) as *const $vec;
                    [$ld(p as *const $cast), $ld(p.add(1) as *const $cast)]
                }
            }

            #[target_feature(enable = $feature)]
            #[inline]
            pub fn $store<$($g)*>(a: &mut [$elem], i: usize, v: $vec) {
                debug_assert!(fits(a, i, size_of::<$vec>()));
                // SAFETY: in bounds by construction (see the module).
                unsafe { $st(a.as_mut_ptr().add(i) as *mut $cast, v) }
            }
        };
    }

    access!("sse2", load_si128, load2_si128, store_si128, __m128i, [T: Plain] T, _mm_loadu_si128, _mm_storeu_si128, __m128i);
    access!("avx", load_si256, load2_si256, store_si256, __m256i, [T: Plain] T, _mm256_loadu_si256, _mm256_storeu_si256, __m256i);
    access!("avx512f", load_si512, load2_si512, store_si512, __m512i, [T: Plain] T, _mm512_loadu_si512, _mm512_storeu_si512, __m512i);
    access!("avx", load_pd, load2_pd, store_pd, __m256d, [] f64, _mm256_loadu_pd, _mm256_storeu_pd, f64);
    access!("avx512f", load_pd8, load2_pd8, store_pd8, __m512d, [] f64, _mm512_loadu_pd, _mm512_storeu_pd, f64);

    /// `a[i..i + 4] = v` in the lanes whose `mask` has the sign bit set.
    #[target_feature(enable = "avx")]
    #[inline]
    pub fn maskstore_pd(a: &mut [f64], i: usize, mask: __m256i, v: __m256d) {
        debug_assert!(i + 4 <= a.len());
        // SAFETY: in bounds by construction (see the module).
        unsafe { _mm256_maskstore_pd(a.as_mut_ptr().add(i), mask, v) }
    }

    /// `a[i..i + 8] = v` in the lanes set in `mask`.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub fn mask_store_pd8(a: &mut [f64], i: usize, mask: __mmask8, v: __m512d) {
        debug_assert!(i + 8 <= a.len());
        // SAFETY: in bounds by construction (see the module).
        unsafe { _mm512_mask_storeu_pd(a.as_mut_ptr().add(i), mask, v) }
    }
}
