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

/// 実行環境を確認済みのハンドル。内積ごとの CPU 検出や enum 分岐はない。
#[derive(Clone, Copy, Debug)]
pub struct ResolvedKernel {
    name: &'static str,
    dot: Dot,
    rows: Rows,
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
    pub(crate) fn dot_validated(self, left: &[f32], right: &[f32]) -> f32 {
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
    pub(crate) fn dot_rows_validated(self, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
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
            });
        }
        if requested == Kernel::Sse2 && std::is_x86_feature_detected!("sse2") {
            return Some(ResolvedKernel {
                name: "sse2",
                dot: x86::dot_sse2,
                rows: x86::rows_sse2,
            });
        }
    }
    #[cfg(target_arch = "aarch64")]
    if requested == Kernel::Neon && std::arch::is_aarch64_feature_detected!("neon") {
        return Some(ResolvedKernel {
            name: "neon",
            dot: neon::dot_neon,
            rows: neon::rows_neon,
        });
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if requested == Kernel::Simd128 {
        return Some(ResolvedKernel {
            name: "simd128",
            dot: wasm::dot_simd128,
            rows: wasm::rows_simd128,
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
}
