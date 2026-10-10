use tzolkin_inference::kernel::{Kernel, ResolvedKernel};

fn kernels() -> Vec<ResolvedKernel> {
    [
        Kernel::Scalar,
        Kernel::Auto,
        Kernel::Avx2,
        Kernel::Sse2,
        Kernel::Neon,
        Kernel::Simd128,
    ]
    .into_iter()
    .filter_map(|kernel| kernel.resolve().ok())
    .collect()
}

fn ordered(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0_f32, |sum, (a, b)| sum + a * b)
}

#[repr(align(64))]
struct Aligned([f32; 576]);

#[test]
fn auto_is_resolved_once_and_explicit_unsupported_backend_is_an_error() {
    assert_eq!(Kernel::default(), Kernel::Scalar);
    let automatic = Kernel::Auto.resolve().unwrap();
    assert_eq!(
        automatic.backend(),
        Kernel::Auto.resolve().unwrap().backend()
    );
    assert_eq!(Kernel::Scalar.resolve().unwrap().backend(), "scalar");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let expected = if std::is_x86_feature_detected!("avx2") {
            "avx2"
        } else if std::is_x86_feature_detected!("sse2") {
            "sse2"
        } else {
            "scalar"
        };
        assert_eq!(automatic.backend(), expected);
        assert!(Kernel::Neon.resolve().is_err());
        assert!(Kernel::Simd128.resolve().is_err());
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert!(Kernel::Avx2.resolve().is_err());
        assert!(Kernel::Sse2.resolve().is_err());
    }
    #[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
    {
        assert_eq!(automatic.backend(), "scalar");
        assert!(Kernel::Simd128.resolve().is_err());
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    assert_eq!(automatic.backend(), "simd128");
}

#[test]
fn every_length_and_unaligned_offset_has_exact_tail_and_no_padding_read() {
    // 先頭は64byte整列。異なるoffsetにより双方の非整列ロードと全tailを網羅。
    let mut left = Aligned([f32::NAN; 576]);
    let mut right = Aligned([f32::INFINITY; 576]);
    for backend in kernels() {
        for length in 0..=513 {
            for offset in 0..16 {
                let other_offset = (offset * 7 + 3) % 16;
                left.0[offset..offset + length].fill(1.0);
                right.0[other_offset..other_offset + length].fill(0.25);
                let a = &left.0[offset..offset + length];
                let b = &right.0[other_offset..other_offset + length];
                let value = backend.dot(a, b).unwrap();
                assert_eq!(
                    value,
                    length as f32 * 0.25,
                    "{} len={length} offset={offset}",
                    backend.backend()
                );
                assert_eq!(value, backend.dot_validated(a, b));
                left.0[offset..offset + length].fill(f32::NAN);
                right.0[other_offset..other_offset + length].fill(f32::INFINITY);
            }
        }
    }
}

#[test]
fn scalar_order_is_bit_exact_and_simd_matches_bounded_model_scale_vectors() {
    let left: Vec<f32> = (0..529)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 128.0)
        .collect();
    let right: Vec<f32> = (0..529)
        .map(|i| ((i * 29 % 97) as f32 - 48.0) / 96.0)
        .collect();
    for length in 0..=513 {
        let a = &left[5..5 + length];
        let b = &right[11..11 + length];
        let reference = ordered(a, b);
        assert_eq!(
            Kernel::Scalar
                .resolve()
                .unwrap()
                .dot(a, b)
                .unwrap()
                .to_bits(),
            reference.to_bits()
        );
        let ideal: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        for backend in kernels() {
            let actual = f64::from(backend.dot(a, b).unwrap());
            assert!(
                (actual - ideal).abs() <= 0.00002 * (1.0 + ideal.abs()),
                "{} len={length}: {actual} vs {ideal}",
                backend.backend()
            );
        }
    }
}

#[test]
fn cancellation_uses_an_absolute_forward_error_bound_not_relative_only() {
    let left: Vec<f32> = [100_000_000.0_f32, 1.0, -100_000_000.0, 1.0].repeat(128);
    let right = vec![1.0; left.len()];
    let ideal: f64 = left.iter().map(|x| f64::from(*x)).sum();
    let absolute_products: f64 = left.iter().map(|x| f64::from(*x).abs()).sum();
    let error_bound = 2.0 * (left.len() + 1) as f64 * f64::from(f32::EPSILON) * absolute_products;
    let scalar = Kernel::Scalar
        .resolve()
        .unwrap()
        .dot(&left, &right)
        .unwrap();
    assert_eq!(scalar.to_bits(), ordered(&left, &right).to_bits());
    for backend in kernels() {
        let actual = f64::from(backend.dot(&left, &right).unwrap());
        assert!(
            (actual - ideal).abs() <= error_bound,
            "{} cancellation bound",
            backend.backend()
        );
    }
}

#[test]
fn invalid_shape_nonfinite_operands_and_overflow_are_rejected() {
    for backend in kernels() {
        assert!(backend.dot(&[1.0], &[]).is_err());
        assert!(backend.dot(&[], &[1.0]).is_err());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(backend.dot(&[invalid], &[0.0]).is_err());
            assert!(backend.dot(&[1.0], &[invalid]).is_err());
        }
        assert!(backend.dot(&[f32::MAX; 33], &[2.0; 33]).is_err());
        assert!(backend.dot(&[f32::MAX; 33], &[1.0; 33]).is_err());
        let mut output = [7.0, 8.0];
        assert!(backend.dot_rows(&[1.0], &[1.0], &mut output).is_err());
        assert_eq!(output, [7.0, 8.0]);
        assert!(
            backend
                .dot_rows(&[f32::NAN, 1.0], &[1.0], &mut output)
                .is_err()
        );
        assert_eq!(output, [7.0, 8.0]);
        assert!(
            backend
                .dot_rows(&[f32::MAX; 2], &[2.0], &mut output)
                .is_err()
        );
    }
}

#[test]
fn matrix_dispatch_matches_each_dot_including_zero_columns_and_512_by_32() {
    for backend in kernels() {
        for columns in [0, 1, 3, 4, 7, 8, 9, 31, 32, 33, 511, 512, 513] {
            let vector: Vec<f32> = (0..columns).map(|i| (i % 17) as f32 / 17.0).collect();
            let matrix: Vec<f32> = (0..columns * 32)
                .map(|i| ((i * 13 % 31) as f32 - 15.0) / 64.0)
                .collect();
            let mut output = [f32::NAN; 32];
            backend.dot_rows(&matrix, &vector, &mut output).unwrap();
            let mut hot_output = [f32::NAN; 32];
            backend.dot_rows_validated(&matrix, &vector, &mut hot_output);
            for row in 0..32 {
                assert_eq!(output[row].to_bits(), hot_output[row].to_bits());
                assert_eq!(
                    output[row].to_bits(),
                    backend
                        .dot(&matrix[row * columns..(row + 1) * columns], &vector)
                        .unwrap()
                        .to_bits()
                );
            }
        }
        let mut no_rows = [];
        backend.dot_rows(&[], &[1.0, 2.0], &mut no_rows).unwrap();
        backend.dot_rows(&[], &[], &mut no_rows).unwrap();
    }
}

#[test]
fn validated_hot_shape_guard_also_runs_in_release() {
    let backend = Kernel::Auto.resolve().unwrap();
    assert!(std::panic::catch_unwind(|| backend.dot_validated(&[1.0], &[])).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            backend.dot_rows_validated(&[1.0], &[1.0], &mut [0.0; 2]);
        })
        .is_err()
    );
}

#[test]
fn unfinished_prefix_continuation_matches_full_dot_bits_with_unaligned_inputs() {
    // Offsets 0..8 cover all SIMD alignments without adding an alignment requirement.
    for offset in 0..8 {
        let mut matrix = vec![0.0; 32 * 512 + offset];
        let mut vector = vec![0.0; 512 + offset];
        for (i, value) in matrix[offset..].iter_mut().enumerate() {
            *value = ((i * 13 % 31) as f32 - 15.0) / 64.0;
        }
        for (i, value) in vector[offset..].iter_mut().enumerate() {
            *value = match i % 19 {
                0 => -0.0,
                1 => f32::from_bits(1), // subnormal
                _ => ((i % 17) as f32 - 8.0) / 17.0,
            };
        }
        let matrix = &matrix[offset..];
        let vector = &mut vector[offset..];
        for backend in kernels() {
            let prefix = backend.policy_prefix_validated(matrix, &vector[..384]);
            // Repeated suffixes share a prefix but vary the candidate contribution.
            for shift in [0.0, 0.25, -0.5] {
                for (i, value) in vector[384..].iter_mut().enumerate() {
                    *value = (i as f32 - 64.0) / 128.0 + shift;
                }
                let mut actual = [0.0; 32];
                prefix.continue_validated(&vector[384..], &mut actual);
                let mut expected = [0.0; 32];
                backend.dot_rows_validated(matrix, vector, &mut expected);
                assert_eq!(
                    actual.map(f32::to_bits),
                    expected.map(f32::to_bits),
                    "{} offset{offset} shift{shift}",
                    backend.backend()
                );
            }
        }
    }
}

#[test]
fn prefix_does_not_reduce_lanes_early_or_hide_nonfinite_final_affine() {
    for backend in kernels() {
        // SIMD prefix horizontal sum overflows, but each unreduced lane can
        // cancel in the suffix. An early reduction would lose this result.
        let mut matrix = vec![0.0; 32 * 512];
        let vector = [1.0; 512];
        for row in matrix.as_chunks_mut::<512>().0.iter_mut() {
            row[..8].fill(f32::MAX / 4.0);
            row[384..392].fill(-f32::MAX / 4.0);
        }
        let cache = backend.policy_prefix_validated(&matrix, &vector[..384]);
        let mut expected = [0.0; 32];
        let mut actual = [0.0; 32];
        backend.dot_rows_validated(&matrix, &vector, &mut expected);
        cache.continue_validated(&vector[384..], &mut actual);
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
        if backend.backend() != "scalar" {
            assert!(actual.iter().all(|v| v.is_finite()));
        } else {
            assert!(actual.iter().all(|v| !v.is_finite()));
        }
        // Finite operands can overflow inside a lane. Retain that nonfinite
        // result for the model's full-affine finite check, never tanh it away.
        matrix.fill(f32::MAX);
        let vector = [2.0; 512];
        let cache = backend.policy_prefix_validated(&matrix, &vector[..384]);
        cache.continue_validated(&vector[384..], &mut actual);
        assert!(actual.iter().all(|v| !v.is_finite()));
        backend.dot_rows_validated(&matrix, &vector, &mut expected);
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }
}

#[test]
fn prefix_shape_guards_hold_before_any_backend_load() {
    for backend in kernels() {
        let matrix = vec![0.0; 32 * 512];
        assert!(
            std::panic::catch_unwind(
                || backend.policy_prefix_validated(&matrix[..matrix.len() - 1], &[0.0; 384])
            )
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| backend.policy_prefix_validated(&matrix, &[0.0; 383]))
                .is_err()
        );
        let cache = backend.policy_prefix_validated(&matrix, &[0.0; 384]);
        assert!(
            std::panic::catch_unwind(|| cache.continue_validated(&[0.0; 127], &mut [0.0; 32]))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| cache.continue_validated(&[0.0; 128], &mut [0.0; 31]))
                .is_err()
        );
    }
}
