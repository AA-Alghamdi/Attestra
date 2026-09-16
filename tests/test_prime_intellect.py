"""Tests for the Prime Intellect GPU provider integration.

Verifies:
1. Key resolution logic (env var, file fallback)
2. PrimeIntellectPodProvider gating (no key = unavailable, no spend)
3. Provider capabilities and probe behavior
4. Build script construction (self-contained, correct VFRESULT format)
"""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from vfplatform.providers import PrimeIntellectPodProvider, ResourceGated
from vfplatform._prime_intellect import resolve_pi_key


class TestKeyResolution:
    def test_explicit_key(self):
        assert resolve_pi_key("my-key-123") == "my-key-123"

    def test_env_var(self):
        with patch.dict(os.environ, {"PRIME_INTELLECT_API_KEY": "env-key-456"}):
            assert resolve_pi_key() == "env-key-456"

    def test_key_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as f:
            f.write("file-key-789\n")
            path = f.name
        try:
            with patch.dict(os.environ, {"VF_PI_KEY_FILE": path}, clear=False):
                # clear the env var to test file fallback
                env = dict(os.environ)
                env.pop("PRIME_INTELLECT_API_KEY", None)
                with patch.dict(os.environ, env, clear=True):
                    assert resolve_pi_key() == "file-key-789"
        finally:
            os.unlink(path)

    def test_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("vfplatform._prime_intellect._key_file", return_value="/nonexistent"):
                assert resolve_pi_key() is None


class TestPrimeIntellectProvider:
    def test_unavailable_without_key(self):
        """No key = not available, honest gating."""
        with patch("vfplatform._prime_intellect.resolve_pi_key", return_value=None):
            p = PrimeIntellectPodProvider(api_key=None)
            p.api_key = None  # force no key
            assert p.available() is False
            caps = p.capabilities()
            assert caps["gated"] is True
            assert "PRIME_INTELLECT_API_KEY" in caps["reason"]

    def test_available_with_key(self):
        """With a key, provider is available."""
        with patch("vfplatform._prime_intellect.resolve_pi_key", return_value="test-key"):
            p = PrimeIntellectPodProvider(api_key="test-key")
            assert p.available() is True
            caps = p.capabilities()
            assert caps["gated"] is False
            assert caps["device"] == "gpu"

    def test_probe_without_key(self):
        with patch("vfplatform._prime_intellect.resolve_pi_key", return_value=None):
            p = PrimeIntellectPodProvider(api_key=None)
            p.api_key = None
            probe = p.probe()
            assert probe["available"] is False

    def test_map_raises_resource_gated(self):
        """map() raises ResourceGated when no key is set."""
        with patch("vfplatform._prime_intellect.resolve_pi_key", return_value=None):
            p = PrimeIntellectPodProvider(api_key=None)
            p.api_key = None
            with pytest.raises(ResourceGated):
                p.map(lambda j, o: o, [])

    def test_build_script_format(self):
        """build_script produces a self-contained Python script with VFRESULT output."""
        p = PrimeIntellectPodProvider(api_key="test-key")
        jobspec = {"family": "torch_mlp", "train_X": [[1, 2]], "train_y": [0]}
        script = p.build_script(jobspec)
        assert "VFRESULT" in script
        assert "torch_mlp" in script or "TorchMLP" in script
        assert "_VF_INPUT" in script


class TestProviderIntegration:
    def test_provider_in_first_available(self):
        """PrimeIntellectPodProvider integrates with first_available."""
        from vfplatform.providers import first_available, LocalCpuProvider
        providers = [
            PrimeIntellectPodProvider(api_key="test-key"),
            LocalCpuProvider(),
        ]
        # PI is available (has key), should be first
        p = first_available(providers)
        assert p.name == "prime-intellect-gpu"

    def test_fallback_to_cpu(self):
        """Falls back to CPU when PI has no key."""
        from vfplatform.providers import first_available, LocalCpuProvider
        pi = PrimeIntellectPodProvider(api_key=None)
        pi.api_key = None
        providers = [pi, LocalCpuProvider()]
        p = first_available(providers)
        assert p.name == "local-cpu"
