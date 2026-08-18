from h3_tools.media import resample_indices


def assert_valid(indices, n_src):
    assert all(0 <= i <= n_src - 1 for i in indices)
    assert indices == sorted(indices)  # monotone non-decreasing


def test_identity_at_24fps():
    assert resample_indices(48, 24.0) == list(range(48))


def test_ntsc_23976_keeps_frame_count():
    idx = resample_indices(100, 23.976)
    assert len(idx) == 100
    assert idx[-1] == 99
    assert_valid(idx, 100)


def test_pal_25_drops_frames():
    idx = resample_indices(50, 25.0)  # 2s clip -> 48 frames at 24fps
    assert len(idx) == 48
    assert_valid(idx, 50)


def test_30fps():
    idx = resample_indices(30, 30.0)  # 1s -> 24 frames
    assert len(idx) == 24
    assert idx[0] == 0
    assert_valid(idx, 30)


def test_60fps_picks_alternating():
    idx = resample_indices(60, 60.0)  # 1s -> 24 frames, every 2.5th source frame
    assert len(idx) == 24
    assert idx[:4] == [0, 2, 5, 8]
    assert_valid(idx, 60)


def test_tiny_clip_returns_short_list():
    assert len(resample_indices(3, 24.0)) == 3  # caller rejects < 5


def test_degenerate_inputs():
    assert resample_indices(0, 24.0) == []
    assert resample_indices(10, 0.0) == []
