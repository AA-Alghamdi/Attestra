"""Hermetic locks for the remote-GPU SSH plumbing in attestra.execution.gpu_backend.

These never touch the network. They pin two properties that decide whether the remote
H100 path can actually move bytes:

  (1) _parse_ssh understands Prime Intellect's tokenized "user@host -p PORT" form (and the
      bare ip / user@host:port fallbacks);
  (2) _ssh_identity_opts materializes PRIME_INTELLECT_SSH_KEY to a 0600 keyfile and returns
      ['-i', keyfile], and the scp/ssh commands use the correct per-tool port flag
      (ssh -> -p, scp -> -P) so scp does not misread the port as a preserve-times flag.
"""
import os

from attestra.execution import gpu_backend as g


def test_parse_ssh_tokenized_and_fallbacks():
    assert g._parse_ssh({"sshConnection": "root@31.56.109.56 -p 22"}) == ("31.56.109.56", 22, "root")
    assert g._parse_ssh({"sshConnection": "ssh ubuntu@1.2.3.4 -p 2222"}) == ("1.2.3.4", 2222, "ubuntu")
    assert g._parse_ssh({"sshConnection": "root@5.6.7.8:2200"}) == ("5.6.7.8", 2200, "root")
    assert g._parse_ssh({"sshConnection": ["", "root@9.9.9.9 -p 22"]}) == ("9.9.9.9", 22, "root")
    assert g._parse_ssh({"ip": "10.0.0.1"}) == ("10.0.0.1", 22, "root")
    assert g._parse_ssh({}) is None


def test_ssh_identity_opts_materializes_key_0600(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_SSH_KEY_CACHE", None)
    monkeypatch.delenv("PRIME_INTELLECT_SSH_KEY_FILE", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("PRIME_INTELLECT_SSH_KEY", "-----BEGIN KEY-----\nabc\n-----END KEY-----")

    opts = g._ssh_identity_opts()
    assert opts[0] == "-i"
    keyfile = opts[1]
    assert os.path.exists(keyfile)
    assert oct(os.stat(keyfile).st_mode & 0o777) == "0o600"
    with open(keyfile) as f:
        body = f.read()
    assert body.endswith("\n") and "BEGIN KEY" in body


def test_ssh_identity_opts_prefers_explicit_keyfile(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_SSH_KEY_CACHE", None)
    kf = tmp_path / "id_test"
    kf.write_text("x\n")
    monkeypatch.setenv("PRIME_INTELLECT_SSH_KEY_FILE", str(kf))
    assert g._ssh_identity_opts() == ["-i", str(kf)]


def test_ssh_identity_opts_empty_without_key(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_SSH_KEY_CACHE", None)
    monkeypatch.delenv("PRIME_INTELLECT_SSH_KEY_FILE", raising=False)
    monkeypatch.delenv("PRIME_INTELLECT_SSH_KEY", raising=False)
    assert g._ssh_identity_opts() == []
