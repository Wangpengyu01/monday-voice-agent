import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from llm import client  # noqa: E402


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "needs Unix sockets")
class LlmClientConnectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = os.path.join(self.tmp.name, "llm.sock")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_missing_socket_raises_unavailable_after_retry(self) -> None:
        t0 = time.monotonic()
        with self.assertRaises(client.LlmUnavailableError):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                client._connect(s, self.sock_path, retry_sec=0.5)
        self.assertGreaterEqual(time.monotonic() - t0, 0.4)

    def test_socket_that_appears_during_retry_is_used(self) -> None:
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        def bring_up_later() -> None:
            time.sleep(0.4)
            server.bind(self.sock_path)
            server.listen(1)
            conn, _ = server.accept()
            conn.recv(4096)
            conn.sendall(b'{"type":"token","id":"t1","text":"\xe5\xa5\xbd"}\n')
            conn.sendall(b'{"type":"done","id":"t1","finish_reason":"stop"}\n')
            conn.close()

        th = threading.Thread(target=bring_up_later, daemon=True)
        th.start()
        try:
            reply = client._read_chat_stream(self.sock_path, {"id": "t1", "type": "chat", "prompt": "x"})
        finally:
            th.join(timeout=5)
            server.close()
        self.assertEqual(reply["text"], "好")
        self.assertEqual(reply["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
