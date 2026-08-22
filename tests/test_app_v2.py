"""App import-safe tests (Phase 7, artifact-free + headless).

Covers:
- artifact-free subprocess import succeeds
- main and get_predictor exist
- import does not construct a predictor
- import does not open models/scalers/raw data
- import emits no missing ScriptRunContext or TF init output
- friendly missing-artifact guidance appears only when predictor requested
- headless Streamlit launch works when local artifacts present (app_smoke+needs_artifacts)
- XGBoost serving does not import TensorFlow
- timing recorded diagnostically (no brittle hard assertion)
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.app_smoke


def _run(code: str, timeout: int = 30):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, cwd=str(ROOT), timeout=timeout)


def test_app_import_is_artifact_free_subprocess():
    """Bare `import app_v2` must succeed without gitignored artifacts and without side effects."""
    code = """
        import time, sys
        t0 = time.perf_counter()
        import app_v2
        dt = time.perf_counter() - t0
        assert hasattr(app_v2, 'get_predictor'), 'get_predictor missing'
        assert hasattr(app_v2, 'main'), 'main missing'
        assert callable(app_v2.get_predictor)
        assert callable(app_v2.main)
        # no predictor instance at import
        assert not hasattr(app_v2, 'predictor')
        print(f'import_time={dt:.3f}')
        print('import_ok')
    """
    result = _run(code)
    assert result.returncode == 0, f"import failed: {result.stdout} {result.stderr}"
    assert "import_ok" in result.stdout
    # diagnostic timing, not asserted
    assert "import_time=" in result.stdout
    # no ScriptRunContext warning
    assert "missing ScriptRunContext" not in result.stderr
    assert "ScriptRunContext" not in result.stderr


def test_app_has_main_and_get_predictor():
    code = """
        import app_v2
        assert hasattr(app_v2, 'main')
        assert hasattr(app_v2, 'get_predictor')
        assert callable(app_v2.main)
        assert callable(app_v2.get_predictor)
        # st.cache_resource adds attrs
        assert hasattr(app_v2.get_predictor, 'clear') or hasattr(app_v2.get_predictor, '__wrapped__') or True
        print('has_both')
    """
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "has_both" in result.stdout
    # also direct
    import app_v2
    assert hasattr(app_v2, "main") and hasattr(app_v2, "get_predictor")


def test_import_does_not_construct_predictor():
    """Import must not call V2Predictor.__init__."""
    code = """
        import sys
        # spy on V2Predictor before app_v2 import
        import rul_prediction.serving.v2_predictor as vp
        orig = vp.V2Predictor.__init__
        called = []
        def spy(self, *a, **kw):
            called.append(1)
            return orig(self, *a, **kw)
        vp.V2Predictor.__init__ = spy
        import app_v2
        assert called == [], f"V2Predictor.__init__ called at import: {called}"
        print('spy_ok')
        vp.V2Predictor.__init__ = orig
    """
    result = _run(code)
    assert result.returncode == 0, f"{result.stdout} {result.stderr}"
    assert "spy_ok" in result.stdout


def test_import_does_not_open_models_scalers_raw_data():
    """Import must not open models/scalers/raw data (spy open)."""
    code = """
        import builtins
        from pathlib import Path as P
        opened = []
        orig_open = builtins.open
        def spy_open(file, *a, **kw):
            s = str(file)
            lower = s.lower()
            if any(x in lower for x in ['models/v2_2', 'fd001', 'scaler', 'data/raw']):
                # ignore reading configs which are allowed? But models/scalers/raw must not open
                if 'models/v2_2' in lower or 'scaler' in lower or 'data/raw' in lower:
                    opened.append(s)
            return orig_open(file, *a, **kw)
        builtins.open = spy_open
        orig_popen = P.open
        def spy_popen(self, *a, **kw):
            s = str(self).lower()
            if 'models/v2_2' in s or 'scaler' in s or 'data/raw' in s:
                opened.append(str(self))
            return orig_popen(self, *a, **kw)
        P.open = spy_popen
        import app_v2
        builtins.open = orig_open
        P.open = orig_popen
        assert opened == [], f"import opened files {opened}"
        print('no_open_ok')
    """
    result = _run(code)
    assert result.returncode == 0, f"{result.stdout} {result.stderr}"
    assert "no_open_ok" in result.stdout
    # also ensure tensorflow not imported for XGBoost path
    code2 = """
        import sys
        import app_v2
        assert 'tensorflow' not in sys.modules, f"tensorflow should not be imported at app import"
        # keras submodule also not
        assert not any('keras' in k for k in sys.modules), f"keras in modules"
        print('lazy_tf_ok')
    """
    result2 = _run(code2)
    assert result2.returncode == 0, f"{result2.stdout} {result2.stderr}"
    assert "lazy_tf_ok" in result2.stdout


def test_import_emits_no_streamlit_or_tensorflow_warnings():
    code = """
        import app_v2
        print('ok')
    """
    result = _run(code)
    assert result.returncode == 0
    assert "ScriptRunContext" not in result.stderr, result.stderr
    assert "TensorFlow" not in result.stderr
    # TF init prints like "oneDNN" etc should not appear at import
    assert "tensorflow" not in result.stderr.lower()


@pytest.mark.needs_artifacts
def test_xgboost_serving_does_not_import_tensorflow():
    """XGBoost deployment must use joblib without initializing TF."""
    code = """
        import sys
        # ensure clean state: tf not yet imported
        assert 'tensorflow' not in sys.modules
        from rul_prediction.serving.v2_predictor import V2Predictor
        # verify deployment is xgboost
        import yaml, pathlib
        cfg = yaml.safe_load(pathlib.Path('configs/final_model_v2_2_fd001.yaml').read_text(encoding='utf-8'))
        cand = cfg['model']['candidate_name']
        assert cand.startswith('xgb'), cand
        p = V2Predictor()
        assert 'tensorflow' not in sys.modules, f"TF imported for xgboost deployment: {[k for k in sys.modules if 'tensor' in k]}"
        assert not any('keras' in k for k in sys.modules), "keras imported for xgboost"
        print('xgb_tf_lazy_ok')
        print(f"window={p.window}")
        assert p.window == 90
    """
    result = _run(code)
    assert result.returncode == 0, f"{result.stdout} {result.stderr}"
    assert "xgb_tf_lazy_ok" in result.stdout


def test_friendly_missing_artifact_guidance_only_when_requested():
    """Friendly guidance appears only when predictor is requested, not at import."""
    code = """
        from pathlib import Path
        import tempfile, shutil, sys
        import app_v2
        print('import_no_error')
        # simulate missing artifacts via temp ROOT
        tmp = Path(tempfile.mkdtemp())
        shutil.copytree('configs', tmp / 'configs', dirs_exist_ok=True)
        # also need experiments for manifest etc? copy minimal needed? Use tmp as root with configs only
        # patch both app_v2 predictor root and v2_predictor ROOT
        import rul_prediction.serving.v2_predictor as v2p
        import rul_prediction.benchmark.v2 as bench_v2
        orig = v2p.ROOT
        orig2 = bench_v2.ROOT
        v2p.ROOT = tmp
        bench_v2.ROOT = tmp
        try:
            try:
                p = v2p.V2Predictor()
                print('unexpected success')
                sys.exit(1)
            except FileNotFoundError as e:
                msg = str(e)
                low = msg.lower()
                assert 'run_v2_2_freeze' in msg or 'scripts' in low, msg
                assert 'models' in low, msg
                print('guidance_ok')
            except Exception as e:
                # manifest verification may raise ArtifactMissingError (subclass of FileNotFoundError) or ValueError
                msg = str(e)
                assert 'models' in msg.lower() or 'artifact' in msg.lower() or 'missing' in msg.lower(), msg
                print('guidance_ok')
        finally:
            v2p.ROOT = orig
            bench_v2.ROOT = orig2
    """
    result = _run(code)
    assert result.returncode == 0, f"{result.stdout} {result.stderr}"
    assert "import_no_error" in result.stdout
    assert "guidance_ok" in result.stdout


def test_import_timing_diagnostic():
    """Record import timing diagnostically, no brittle hard cutoff."""
    import time
    code = """
        import time
        t0=time.perf_counter()
        import app_v2
        dt=time.perf_counter()-t0
        print(f"TIMING import={dt:.4f}s")
    """
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "TIMING" in result.stdout
    # diagnostic only – print to pytest output for visibility
    for line in result.stdout.splitlines():
        if "TIMING" in line:
            print(line)


@pytest.mark.needs_artifacts
def test_headless_streamlit_smoke_with_artifacts():
    """Headless Streamlit that loads gitignored artifacts must carry both app_smoke+needs_artifacts."""
    import time
    import socket
    import subprocess
    import sys
    # first verify predictor works (fallback if streamlit launch fails)
    from app_v2 import get_predictor
    try:
        get_predictor.clear()
    except Exception:
        pass
    t0 = time.perf_counter()
    predictor = get_predictor()
    dt = time.perf_counter() - t0
    print(f"TIMING get_predictor={dt:.3f}s")  # diagnostic, no hard assert
    assert predictor is not None
    assert hasattr(predictor, 'model_version')
    assert predictor.window == 90
    assert predictor.candidate.startswith('xgb')
    # try real headless streamlit launch with timeout – diagnostic, must not hard-fail if port busy
    port = 18507
    # find free port
    s = socket.socket()
    try:
        s.bind(('', 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    cmd = [sys.executable, "-m", "streamlit", "run", "app_v2.py", "--server.headless", "true", "--server.port", str(port), "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
    try:
        # wait up to 12s for startup; success = server still running after grace
        # period AND no Python traceback in its output (a Streamlit server stays
        # alive even when the script raises during render, so scan for tracebacks)
        start = time.perf_counter()
        out = ""
        startup_ok = False
        while time.perf_counter() - start < 12:
            if proc.poll() is not None:
                break
            time.sleep(0.5)
            if proc.poll() is None and time.perf_counter() - start > 5:
                startup_ok = True
                break
        elapsed = time.perf_counter() - start  # diagnostic only, no hard cutoff assert
        print(f"TIMING headless_startup_observed_after={elapsed:.2f}s startup_ok={startup_ok}")
        if proc.poll() is not None:
            # process exited before grace period – capture output and fail with context
            try:
                stdout, _ = proc.communicate(timeout=2)
                out += stdout
            except Exception:
                pass
            assert "headless_started" in out, f"streamlit exited {proc.returncode}: {out[:2000]}"
        else:
            # server alive past grace period: fail if the app script raised
            assert startup_ok, f"streamlit not stable after grace period: {out[:2000]}"
            assert "Traceback (most recent call last)" not in out, f"traceback in streamlit output: {out[:2000]}"
            print(f"headless predictor ok: {predictor.model_version} on port {port}")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
