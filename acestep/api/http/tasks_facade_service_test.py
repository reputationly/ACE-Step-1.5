"""Unit tests for the GPUStack facade adaptation helpers."""

import os
import tempfile
import unittest
from types import SimpleNamespace

from acestep.api.http.tasks_facade_service import (
    FACADE_CANCELLED,
    atomic_place,
    build_request,
    map_status,
    materialize_output,
)


class TestMapStatus(unittest.TestCase):
    """Store status -> facade status must match the facade _ENGINE_STATE_MAP."""

    def test_known_statuses(self):
        self.assertEqual(map_status("queued"), "pending")
        self.assertEqual(map_status("running"), "processing")
        self.assertEqual(map_status("succeeded"), "completed")
        self.assertEqual(map_status("failed"), "failed")

    def test_unknown_falls_back_to_processing(self):
        # An unexpected value must never look terminal to the facade.
        self.assertEqual(map_status("weird"), "processing")

    def test_cancelled_constant_is_double_l(self):
        self.assertEqual(FACADE_CANCELLED, "cancelled")


class TestAtomicPlace(unittest.TestCase):
    """Atomic copy leaves only the final file and preserves the extension."""

    def test_places_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.mp3")
            with open(src, "wb") as f:
                f.write(b"AUDIO")
            outdir = os.path.join(d, "out")
            dst = os.path.join(outdir, "task.mp3")
            atomic_place(src, dst)
            self.assertTrue(os.path.exists(dst))
            with open(dst, "rb") as f:
                self.assertEqual(f.read(), b"AUDIO")
            # Destination dir holds only the final file — no ``.part`` leftover.
            self.assertEqual(os.listdir(outdir), ["task.mp3"])

    def test_missing_source_raises_and_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "out")
            dst = os.path.join(outdir, "task.mp3")
            with self.assertRaises(OSError):
                atomic_place(os.path.join(d, "nope.mp3"), dst)
            self.assertFalse(os.path.exists(dst))
            # No half-written temp left behind on failure.
            self.assertEqual(os.listdir(outdir), [])


class TestMaterializeOutput(unittest.TestCase):
    """Materialize the first raw audio path onto the injected NFS path."""

    def _record(self, result):
        return SimpleNamespace(result=result)

    def test_success(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "gen.mp3")
            with open(src, "wb") as f:
                f.write(b"X")
            dst = os.path.join(d, "nfs", "t.mp3")
            ok, err = materialize_output(self._record({"raw_audio_paths": [src]}), dst)
            self.assertTrue(ok)
            self.assertIsNone(err)
            self.assertTrue(os.path.exists(dst))

    def test_no_audio(self):
        ok, err = materialize_output(self._record({"raw_audio_paths": []}), "/tmp/x.mp3")
        self.assertFalse(ok)
        self.assertIn("no audio", err)

    def test_missing_engine_file(self):
        ok, err = materialize_output(
            self._record({"raw_audio_paths": ["/does/not/exist.mp3"]}), "/tmp/x.mp3"
        )
        self.assertFalse(ok)
        self.assertIn("missing", err)

    def test_missing_save_path(self):
        ok, err = materialize_output(self._record({"raw_audio_paths": ["/a.mp3"]}), "")
        self.assertFalse(ok)
        self.assertIn("save_result_path", err)

    def test_output_root_allows_path_inside(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "gen.mp3")
            with open(src, "wb") as f:
                f.write(b"X")
            root = os.path.join(d, "out")
            dst = os.path.join(root, "u", "t.mp3")
            ok, err = materialize_output(
                self._record({"raw_audio_paths": [src]}), dst, output_root=root
            )
            self.assertTrue(ok, err)
            self.assertTrue(os.path.exists(dst))

    def test_output_root_rejects_escape(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "gen.mp3")
            with open(src, "wb") as f:
                f.write(b"X")
            root = os.path.join(d, "out")
            os.makedirs(root)
            escape = os.path.join(root, "..", "evil.mp3")  # resolves outside root
            ok, err = materialize_output(
                self._record({"raw_audio_paths": [src]}), escape, output_root=root
            )
            self.assertFalse(ok)
            self.assertIn("escapes output root", err)
            self.assertFalse(os.path.exists(os.path.join(d, "evil.mp3")))


class TestBuildRequest(unittest.TestCase):
    """Facade body -> GenerateMusicRequest: strip facade keys, derive format."""

    def _cls(self):
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        return factory, captured

    def test_strips_facade_keys_and_derives_format(self):
        factory, captured = self._cls()
        body = {
            "save_result_path": "/nfs/out/t2m-x/2026/07/18/1/abc.mp3",
            "task_id": "abc",
            "prompt": "an upbeat pop track",
            "lyrics": "[inst]",
            "task_type": "text2music",
        }
        build_request(body, factory)
        self.assertNotIn("save_result_path", captured)
        self.assertNotIn("task_id", captured)
        self.assertEqual(captured["audio_format"], "mp3")
        self.assertEqual(captured["prompt"], "an upbeat pop track")

    def test_explicit_audio_format_not_overridden(self):
        factory, captured = self._cls()
        build_request(
            {"save_result_path": "/nfs/a.mp3", "audio_format": "wav"}, factory
        )
        self.assertEqual(captured["audio_format"], "wav")

    def test_no_extension_leaves_format_unset(self):
        factory, captured = self._cls()
        build_request({"save_result_path": "/nfs/a", "prompt": "x"}, factory)
        self.assertNotIn("audio_format", captured)


try:
    from acestep.api.http.release_task_models import GenerateMusicRequest

    _HAS_REAL_MODEL = True
except Exception:  # noqa: BLE001 - pydantic/model deps may be absent on some envs
    _HAS_REAL_MODEL = False


@unittest.skipUnless(_HAS_REAL_MODEL, "GenerateMusicRequest not importable here")
class TestBuildRequestRealModel(unittest.TestCase):
    """build_request against the real GenerateMusicRequest (facade passthrough)."""

    def test_cover_reference_path_passes_through(self):
        req = build_request(
            {
                "save_result_path": "/nfs/cover-x/2026/07/18/1/a.mp3",
                "task_id": "x",
                "task_type": "cover",
                "prompt": "remix this",
                "reference_audio_path": "/nfs/inputs/ref.wav",
            },
            GenerateMusicRequest,
        )
        self.assertEqual(req.task_type, "cover")
        self.assertEqual(req.reference_audio_path, "/nfs/inputs/ref.wav")
        self.assertEqual(req.audio_format, "mp3")

    def test_repaint_fields_pass_through_and_wav_format(self):
        req = build_request(
            {
                "save_result_path": "/nfs/repaint-x/a.wav",
                "task_type": "repaint",
                "src_audio_path": "/nfs/inputs/src.wav",
                "repainting_start": 5.0,
                "repainting_end": 15.0,
            },
            GenerateMusicRequest,
        )
        self.assertEqual(req.src_audio_path, "/nfs/inputs/src.wav")
        self.assertEqual(req.repainting_start, 5.0)
        self.assertEqual(req.audio_format, "wav")

    def test_unknown_key_is_ignored(self):
        # pydantic v1 (no extra="forbid") silently drops unknown keys.
        req = build_request(
            {"save_result_path": "/n/a.mp3", "bogus_key": 1, "prompt": "p"},
            GenerateMusicRequest,
        )
        self.assertFalse(hasattr(req, "bogus_key"))
        self.assertEqual(req.prompt, "p")


if __name__ == "__main__":
    unittest.main()
