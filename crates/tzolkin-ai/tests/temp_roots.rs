use crate::test_temp_root as temp_root;

#[test]
fn acquisition_preserves_collision_markers_and_returns_other_errors_without_retry() {
    struct Owned(std::path::PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let root = Owned(temp_root::create("tzolkin-temp-boundary").unwrap());
    let occupied = root.0.join("occupied");
    std::fs::create_dir(&occupied).unwrap();
    let marker = occupied.join("marker");
    std::fs::write(&marker, b"foreign bytes").unwrap();
    let mut calls = 0;
    let acquired = temp_root::create_with(&root.0, || {
        calls += 1;
        if calls == 1 { "occupied" } else { "acquired" }.into()
    })
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(acquired, root.0.join("acquired"));
    assert_eq!(std::fs::read(&marker).unwrap(), b"foreign bytes");

    calls = 0;
    let error = temp_root::create_with(&root.0, || {
        calls += 1;
        "occupied".into()
    })
    .unwrap_err();
    assert_eq!(calls, temp_root::MAX_ATTEMPTS);
    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(std::fs::read(&marker).unwrap(), b"foreign bytes");

    let missing_parent = root.0.join("missing");
    let expected = std::fs::create_dir(missing_parent.join("child")).unwrap_err();
    calls = 0;
    let actual = temp_root::create_with(&missing_parent, || {
        calls += 1;
        "child".into()
    })
    .unwrap_err();
    assert_eq!(calls, 1);
    assert_eq!(actual.kind(), expected.kind());
    assert_eq!(actual.raw_os_error(), expected.raw_os_error());
    assert!(!missing_parent.exists());
}
