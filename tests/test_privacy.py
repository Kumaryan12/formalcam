import socket
import subprocess
import sys

import pytest

from formalcam import privacy


def test_unapproved_mediapipe_is_rejected_before_runtime_import(monkeypatch):
    monkeypatch.setattr(privacy, 'version', lambda name: '0.10.35')
    monkeypatch.setattr(privacy, 'restrict_network', lambda: pytest.fail('Version must be checked first'))
    with pytest.raises(RuntimeError, match='not approved'):
        privacy.prepare_pose_runtime()


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS process sandbox')
def test_runtime_sandbox_blocks_tcp_even_to_a_live_local_server():
    # Restrict a child process only, preserving pytest and development tooling.
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        port = server.getsockname()[1]
        script = '''
import socket
import sys
from formalcam.privacy import restrict_network
restrict_network()
try:
    with socket.socket() as client:
        client.settimeout(1.)
        client.connect(('127.0.0.1', int(sys.argv[1])))
except PermissionError:
    print('BLOCKED')
else:
    raise AssertionError('Runtime network connection was allowed')
'''
        result = subprocess.run([sys.executable, '-c', script, str(port)],
                                capture_output=True, text=True, timeout=5.)
    assert result.returncode == 0, result.stderr
    assert 'BLOCKED' in result.stdout
