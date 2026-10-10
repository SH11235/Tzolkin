//! 推論用の内積カーネル。学習の順序付き Scalar 経路とは分離する。
//!
//! SIMD は加算順序が異なるため bit 一致を保証しない。FMA は使用しない。
//! 解決済みハンドルをモデルの推論開始時に作り、行列は一度の dispatch で計算する。
use std::sync::OnceLock;

/// 推論バックエンドの要求。既定値は再現性を優先する Scalar。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kernel {
    #[default]
    Scalar,
    Auto,
    Avx2,
    Sse2,
    Neon,
    Simd128,
}

type Dot = unsafe fn(&[f32], &[f32]) -> f32;
type Rows = unsafe fn(&[f32], &[f32], &mut [f32]);
const PREFIX_COLUMNS: usize = 384;
const POLICY_COLUMNS: usize = 512;
const POLICY_ROWS: usize = 32;
type PrefixState = [[f32; 8]; POLICY_ROWS];
type PreparePrefix = unsafe fn(&[f32], &[f32], &mut PrefixState);
type ContinuePrefix = unsafe fn(&[f32], &[f32], &PrefixState, &mut [f32]);

/// One call's unfinished accumulators, bound to its immutable matrix and backend.
/// No horizontal reduction or bias is performed at the prefix boundary. The
/// caller validates finite operands; continuation accepts only the 128 suffix
/// columns so it cannot accidentally substitute another context.
pub struct PolicyPrefix<'a> {
    matrix: &'a [f32],
    state: PrefixState,
    continuation: ContinuePrefix,
}
impl PolicyPrefix<'_> {
    pub fn continue_validated(&self, suffix: &[f32], output: &mut [f32]) {
        assert_eq!(suffix.len(), POLICY_COLUMNS - PREFIX_COLUMNS);
        assert_eq!(output.len(), POLICY_ROWS);
        // SAFETY: construction checked fixed matrix/prefix shape and resolved
        // CPU support. The checked suffix/output lengths bound every access.
        unsafe { (self.continuation)(self.matrix, suffix, &self.state, output) }
    }
}

/// 実行環境を確認済みのハンドル。内積ごとの CPU 検出や enum 分岐はない。
///
/// The `*_validated` methods are safe, shape-checked numeric primitives. They
/// skip repeated finite-operand scans and may return non-finite accumulators;
/// callers must validate immutable operands once and reject the full affine
/// result after bias. They confer no model, training or source qualification.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedKernel {
    name: &'static str,
    dot: Dot,
    rows: Rows,
    prepare_prefix: PreparePrefix,
    continue_prefix: ContinuePrefix,
}

impl Kernel {
    /// 利用可能な経路を解決する。明示指定が非対応の場合は Scalar へ黙って落とさない。
    pub fn resolve(self) -> Result<ResolvedKernel, String> {
        if self == Self::Auto {
            static AUTO: OnceLock<ResolvedKernel> = OnceLock::new();
            return Ok(*AUTO.get_or_init(|| {
                for requested in [Self::Avx2, Self::Sse2, Self::Neon, Self::Simd128] {
                    if let Some(kernel) = available(requested) {
                        return kernel;
                    }
                }
                scalar()
            }));
        }
        available(self).ok_or_else(|| format!("Kernel {self:?} is not supported by this build/CPU"))
    }
}

impl ResolvedKernel {
    /// Fixed schema-2 hidden layer only. The model boundary has validated finite
    /// parameters and features. Shapes are checked in release before any load.
    /// The saved state is deliberately allowed to be non-finite: only the full
    /// affine result plus bias is subject to the existing finite guard.
    pub fn policy_prefix_validated<'a>(
        self,
        matrix: &'a [f32],
        prefix: &[f32],
    ) -> PolicyPrefix<'a> {
        assert_eq!(matrix.len(), POLICY_COLUMNS * POLICY_ROWS);
        assert_eq!(prefix.len(), PREFIX_COLUMNS);
        let mut state = [[0.0; 8]; POLICY_ROWS];
        // SAFETY: resolved backend and fixed checked dimensions. Prefix length
        // is divisible by every supported SIMD width; there is no partial lane.
        unsafe { (self.prepare_prefix)(matrix, prefix, &mut state) };
        PolicyPrefix {
            matrix,
            state,
            continuation: self.continue_prefix,
        }
    }

    /// 実際に使用するバックエンド名。
    pub fn backend(self) -> &'static str {
        self.name
    }

    /// 外部入力用境界。同長・有限な入力と有限な結果を要求する。
    pub fn dot(self, left: &[f32], right: &[f32]) -> Result<f32, String> {
        if left.len() != right.len() || left.iter().chain(right).any(|value| !value.is_finite()) {
            return Err("Invalid dot operand shape/finite values".into());
        }
        let result = self.dot_validated(left, right);
        if result.is_finite() {
            Ok(result)
        } else {
            Err("Non-finite dot result".into())
        }
    }

    /// 行優先行列とベクトルの内積。行数は output.len()、列数は vector.len()。
    /// 形状・入力エラーでは output を変更しない。結果 overflow は Err となる。
    pub fn dot_rows(
        self,
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        if vector.len().checked_mul(output.len()) != Some(matrix.len())
            || matrix.iter().chain(vector).any(|value| !value.is_finite())
        {
            return Err("Invalid row-dot shape/finite values".into());
        }
        self.dot_rows_validated(matrix, vector, output);
        if output.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err("Non-finite row-dot result".into())
        }
    }

    /// モデル境界で同長・有限値を検証済みの内積。
    ///
    /// shape は release でも再確認し、SIMD の読み出し境界を守る。有限値の再走査は
    /// 行わない。呼出側は bias 加算後を含む非有限な結果を必ず拒否する。
    #[inline]
    pub fn dot_validated(self, left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len(), "Validated dot shape mismatch");
        // SAFETY: resolve が CPU/ビルドの命令対応を確認済み。両 slice の長さは同じ。
        // SIMD は完全な lane の範囲だけ loadu し、残りは安全な scalar indexing を使う。
        unsafe { (self.dot)(left, right) }
    }

    /// モデル検証済みの行列。matrix.len() == vector.len() * output.len()、入力は有限。
    ///
    /// backend 呼出しは行列全体で一回。backend 内の row→dot は直接呼出しなので、
    /// 512×32 hidden 層で 32 回の関数ポインタ dispatch を発生させない。
    /// shape は release でも確認する。非有限な結果の扱いは dot_validated と同じ。
    #[inline]
    pub fn dot_rows_validated(self, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
        assert_eq!(
            vector.len().checked_mul(output.len()),
            Some(matrix.len()),
            "Validated row-dot shape mismatch"
        );
        // SAFETY: resolve が命令対応を確認済み。行列形状の確認により全行が vector と同長。
        // 出力は独立した mutable slice。ゼロ列は backend が load せずゼロで埋める。
        unsafe { (self.rows)(matrix, vector, output) }
    }
}

fn scalar() -> ResolvedKernel {
    ResolvedKernel {
        name: "scalar",
        dot: dot_scalar,
        rows: rows_scalar,
        prepare_prefix: prepare_scalar,
        continue_prefix: continue_scalar,
    }
}

fn available(requested: Kernel) -> Option<ResolvedKernel> {
    if requested == Kernel::Scalar {
        return Some(scalar());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if requested == Kernel::Avx2 && std::is_x86_feature_detected!("avx2") {
            return Some(ResolvedKernel {
                name: "avx2",
                dot: x86::dot_avx2,
                rows: x86::rows_avx2,
                prepare_prefix: x86::prepare_avx2,
                continue_prefix: x86::continue_avx2,
            });
        }
        if requested == Kernel::Sse2 && std::is_x86_feature_detected!("sse2") {
            return Some(ResolvedKernel {
                name: "sse2",
                dot: x86::dot_sse2,
                rows: x86::rows_sse2,
                prepare_prefix: x86::prepare_sse2,
                continue_prefix: x86::continue_sse2,
            });
        }
    }
    #[cfg(target_arch = "aarch64")]
    if requested == Kernel::Neon && std::arch::is_aarch64_feature_detected!("neon") {
        return Some(ResolvedKernel {
            name: "neon",
            dot: neon::dot_neon,
            rows: neon::rows_neon,
            prepare_prefix: neon::prepare_neon,
            continue_prefix: neon::continue_neon,
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if requested == Kernel::Simd128 {
        return Some(ResolvedKernel {
            name: "simd128",
            dot: wasm::dot_simd128,
            rows: wasm::rows_simd128,
            prepare_prefix: wasm::prepare_simd128,
            continue_prefix: wasm::continue_simd128,
        });
    }
    None
}

#[inline]
fn dot_scalar(left: &[f32], right: &[f32]) -> f32 {
    // model::scalar_dot の参照順序と同じ。mul_add や複数 accumulator を使わない。
    left.iter()
        .zip(right)
        .fold(0.0_f32, |sum, (a, b)| sum + a * b)
}

#[inline]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
fn tail(left: &[f32], right: &[f32], start: usize, mut sum: f32) -> f32 {
    for index in start..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

fn rows_scalar(matrix: &[f32], vector: &[f32], output: &mut [f32]) {
    if vector.is_empty() {
        output.fill(0.0);
        return;
    }
    for (row, value) in matrix.chunks_exact(vector.len()).zip(output) {
        *value = dot_scalar(row, vector);
    }
}

fn prepare_scalar(matrix: &[f32], prefix: &[f32], state: &mut PrefixState) {
    for (row, saved) in matrix.as_chunks::<POLICY_COLUMNS>().0.iter().zip(state) {
        saved[0] = dot_scalar(&row[..PREFIX_COLUMNS], prefix);
    }
}
fn continue_scalar(matrix: &[f32], suffix: &[f32], state: &PrefixState, output: &mut [f32]) {
    for ((row, saved), value) in matrix
        .as_chunks::<POLICY_COLUMNS>()
        .0
        .iter()
        .zip(state)
        .zip(output)
    {
        *value = row[PREFIX_COLUMNS..]
            .iter()
            .zip(suffix)
            .fold(saved[0], |sum, (a, b)| sum + a * b);
    }
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! simd_rows {
    ($name:ident, $dot:ident, $feature:literal) => {
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $name(matrix: &[f32], vector: &[f32], output: &mut [f32]) {
            if vector.is_empty() {
                output.fill(0.0);
                return;
            }
            for (row, value) in matrix.chunks_exact(vector.len()).zip(output) {
                // SAFETY: 同じ target_feature の backend 内で直接呼ぶ。形状検証済みで
                // row と vector は同長。output の mutable 借用は入力の借用と重ならない。
                *value = unsafe { $dot(row, vector) };
            }
        }
    };
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! simd_prefix_rows {
    ($prepare:ident, $continue:ident, $feature:literal, $width:literal,
     $zero:expr, $load:ident, $store:ident, $mul:ident, $add:ident) => {
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $prepare(
            matrix: &[f32],
            prefix: &[f32],
            state: &mut super::PrefixState,
        ) {
            // SAFETY: the private factory checks 32x512 and prefix384; resolved
            // feature support is retained by the cache. All loads/stores are
            // unaligned, full-width, and within their respective slices.
            unsafe {
                for (row, saved) in matrix.chunks_exact(super::POLICY_COLUMNS).zip(state) {
                    let mut sum = $zero;
                    for index in (0..super::PREFIX_COLUMNS).step_by($width) {
                        let a = $load(row.as_ptr().add(index));
                        let b = $load(prefix.as_ptr().add(index));
                        sum = $add(sum, $mul(a, b));
                    }
                    $store(saved.as_mut_ptr(), sum);
                }
            }
        }
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $continue(
            matrix: &[f32],
            suffix: &[f32],
            state: &super::PrefixState,
            output: &mut [f32],
        ) {
            // SAFETY: the cache binds this backend and matrix; continuation
            // checks suffix128/output32. Lane phase is unchanged at index384.
            unsafe {
                for ((row, saved), value) in matrix
                    .chunks_exact(super::POLICY_COLUMNS)
                    .zip(state)
                    .zip(output)
                {
                    let mut sum = $load(saved.as_ptr());
                    for index in (0..super::POLICY_COLUMNS - super::PREFIX_COLUMNS).step_by($width)
                    {
                        let a = $load(row.as_ptr().add(super::PREFIX_COLUMNS + index));
                        let b = $load(suffix.as_ptr().add(index));
                        sum = $add(sum, $mul(a, b));
                    }
                    let mut lanes = [0.0_f32; $width];
                    $store(lanes.as_mut_ptr(), sum);
                    // Identical to the existing full-width dot's final reduce.
                    *value = lanes.into_iter().sum();
                }
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::tail;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_avx2(left: &[f32], right: &[f32]) -> f32 {
        // SAFETY: caller は AVX2 と同長を保証。end は両 slice 内の8要素境界。
        // loadu/storeu は16/32byte alignmentを要求せず、各ロードは end を越えない。
        unsafe {
            let end = left.len() / 8 * 8;
            let mut sum = _mm256_setzero_ps();
            let mut index = 0;
            while index < end {
                let a = _mm256_loadu_ps(left.as_ptr().add(index));
                let b = _mm256_loadu_ps(right.as_ptr().add(index));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                index += 8;
            }
            let mut lanes = [0.0_f32; 8];
            _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
            tail(left, right, end, lanes.into_iter().sum())
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn dot_sse2(left: &[f32], right: &[f32]) -> f32 {
        // SAFETY: caller は SSE2 と同長を保証。完全な4要素だけ loadu し、tailはscalar。
        // storeu先は4要素の独立した stack 配列で、alignment 要求はない。
        unsafe {
            let end = left.len() / 4 * 4;
            let mut sum = _mm_setzero_ps();
            let mut index = 0;
            while index < end {
                let a = _mm_loadu_ps(left.as_ptr().add(index));
                let b = _mm_loadu_ps(right.as_ptr().add(index));
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
                index += 4;
            }
            let mut lanes = [0.0_f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), sum);
            tail(left, right, end, lanes.into_iter().sum())
        }
    }

    simd_rows!(rows_avx2, dot_avx2, "avx2");
    simd_rows!(rows_sse2, dot_sse2, "sse2");
    simd_prefix_rows!(
        prepare_avx2,
        continue_avx2,
        "avx2",
        8,
        _mm256_setzero_ps(),
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_mul_ps,
        _mm256_add_ps
    );
    simd_prefix_rows!(
        prepare_sse2,
        continue_sse2,
        "sse2",
        4,
        _mm_setzero_ps(),
        _mm_loadu_ps,
        _mm_storeu_ps,
        _mm_mul_ps,
        _mm_add_ps
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::tail;
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot_neon(left: &[f32], right: &[f32]) -> f32 {
        // SAFETY: caller は NEON と同長を保証。完全な4要素だけロードし、残りはscalar。
        // vld1q/vst1q は f32 slice の alignment で利用できる。FMAは使用しない。
        unsafe {
            let end = left.len() / 4 * 4;
            let mut sum = vdupq_n_f32(0.0);
            let mut index = 0;
            while index < end {
                let a = vld1q_f32(left.as_ptr().add(index));
                let b = vld1q_f32(right.as_ptr().add(index));
                sum = vaddq_f32(sum, vmulq_f32(a, b));
                index += 4;
            }
            let mut lanes = [0.0_f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), sum);
            tail(left, right, end, lanes.into_iter().sum())
        }
    }
    simd_rows!(rows_neon, dot_neon, "neon");
    simd_prefix_rows!(
        prepare_neon,
        continue_neon,
        "neon",
        4,
        vdupq_n_f32(0.0),
        vld1q_f32,
        vst1q_f32,
        vmulq_f32,
        vaddq_f32
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::tail;
    use std::arch::wasm32::*;

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(super) unsafe fn dot_simd128(left: &[f32], right: &[f32]) -> f32 {
        // SAFETY: SIMD128対応ビルド内でのみ生成。ロードは同長sliceの完全な4要素内。
        // v128_load/store はunaligned可能で、store先も16byteのstack配列内。
        unsafe {
            let end = left.len() / 4 * 4;
            let mut sum = f32x4_splat(0.0);
            let mut index = 0;
            while index < end {
                let a = v128_load(left.as_ptr().add(index).cast());
                let b = v128_load(right.as_ptr().add(index).cast());
                sum = f32x4_add(sum, f32x4_mul(a, b));
                index += 4;
            }
            let mut lanes = [0.0_f32; 4];
            v128_store(lanes.as_mut_ptr().cast(), sum);
            tail(left, right, end, lanes.into_iter().sum())
        }
    }
    simd_rows!(rows_simd128, dot_simd128, "simd128");
    #[inline]
    unsafe fn load_f32(pointer: *const f32) -> v128 {
        // SAFETY: macro callers check a complete four-f32 lane.
        unsafe { v128_load(pointer.cast()) }
    }
    #[inline]
    unsafe fn store_f32(pointer: *mut f32, value: v128) {
        // SAFETY: macro callers supply at least four writable f32 values.
        unsafe { v128_store(pointer.cast(), value) }
    }
    simd_prefix_rows!(
        prepare_simd128,
        continue_simd128,
        "simd128",
        4,
        f32x4_splat(0.0),
        load_f32,
        store_f32,
        f32x4_mul,
        f32x4_add
    );
}
