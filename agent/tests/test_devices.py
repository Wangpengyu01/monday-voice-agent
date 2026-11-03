import http.client
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.fernet import Fernet  # noqa: E402

from devices import pairing  # noqa: E402
from devices.inventory import Inventory  # noqa: E402
from devices.scanner import Device, classify  # noqa: E402
from devices.vault import Vault, VaultLocked  # noqa: E402
from devices.voice import describe, parse_command, spoken_digits  # noqa: E402


class ClassifyTests(unittest.TestCase):
    def test_kinds_seen_on_the_real_lan(self) -> None:
        def kind(**kw):
            return classify(Device(ip=kw.pop("ip", "10.0.0.9"), **kw), gateway="10.0.0.1")
        self.assertEqual(kind(ip="10.0.0.1", ports=[22, 53, 80]), "router")
        self.assertEqual(kind(ports=[22, 135, 139, 443, 445, 3389]), "windows-pc")
        self.assertEqual(kind(ports=[22, 5000], services=["iTunes Remote Control"]), "mac")
        self.assertEqual(kind(ports=[53, 5000, 49152, 62078], services=["_meshcop._udp"]), "apple-hub")
        self.assertEqual(kind(ports=[49152, 62078]), "phone")
        self.assertEqual(kind(ports=[80], hostname="tenda.lan"), "ap")
        self.assertEqual(kind(ports=[22], hostname="NanoPi-M6.lan"), "linux")
        self.assertEqual(kind(mac="3a:f3:f4:71:58:c3"), "phone")  # randomized MAC
        self.assertEqual(kind(ports=[5000, 5001]), "nas")


class VoiceTests(unittest.TestCase):
    def test_commands(self) -> None:
        self.assertEqual(parse_command("扫描一下家里的设备"), "scan")
        self.assertEqual(parse_command("家里有哪些设备"), "list")
        self.assertEqual(parse_command("设备配对码"), "pair")
        self.assertEqual(parse_command("我要填设备密码"), "pair")
        self.assertIsNone(parse_command("今天天气怎么样"))

    def test_spoken(self) -> None:
        self.assertEqual(spoken_digits("10.0.0.177"), "一零点零点零点一七七")
        ds = [Device(ip="1", kind="windows-pc"), Device(ip="2", kind="windows-pc"), Device(ip="3", kind="phone"),
              Device(ip="4", kind="unknown")]
        self.assertEqual(describe(ds), "两台Windows 电脑，还有一部手机，另外一台没认出来")


class VaultTests(unittest.TestCase):
    def test_roundtrip_encrypted_and_locked_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            key = Fernet.generate_key()
            v = Vault(Path(d) / "v.db", key=key)
            v.put("aa:bb", "ssh", "pi", "s3cret-pass")
            self.assertEqual(v.get("aa:bb", "ssh").secret, "s3cret-pass")
            self.assertEqual(v.describe(), {"aa:bb": ["ssh"]})
            self.assertNotIn(b"s3cret-pass", (Path(d) / "v.db").read_bytes())
            locked = Vault(Path(d) / "v.db", key="")
            self.assertFalse(locked.unlocked)
            with self.assertRaises(VaultLocked):
                locked.get("aa:bb", "ssh")
            with self.assertRaises(VaultLocked):
                Vault(Path(d) / "v.db", key=Fernet.generate_key()).get("aa:bb", "ssh")
            v._db.close()
            locked._db.close()


class PairingAndInventoryTests(unittest.TestCase):
    def test_code_is_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            code = pairing.new_code(Path(d) / "p.json")
            self.assertFalse(pairing.check_code(Path(d) / "p.json", "x"))
            self.assertTrue(pairing.check_code(Path(d) / "p.json", code))
            self.assertFalse(pairing.check_code(Path(d) / "p.json", code))

    def test_inventory_reports_new_devices_once_and_keeps_nicknames(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inv = Inventory(Path(d) / "d.db")
            dev = Device(ip="10.0.0.155", mac="d8:5e", hostname="DESKTOP-B.lan", kind="windows-pc")
            self.assertEqual(len(inv.update([dev])), 1)
            inv.set_nickname("d8:5e", "我的电脑")
            self.assertEqual(inv.update([dev]), [])
            self.assertEqual(inv.find("帮我打开我的电脑").ip, "10.0.0.155")
            inv._db.close()


class WebTests(unittest.TestCase):
    def test_pairing_gate_and_storing_a_password(self) -> None:
        from devices import web
        with tempfile.TemporaryDirectory() as d:
            web.RUN = Path(d)
            st = web.State()
            st.vault = Vault(Path(d) / "vault.db", key=Fernet.generate_key())
            st.inventory.update([Device(ip="10.0.0.237", mac="e0:0a", kind="linux")])
            web.STATE = st
            srv = web.Server(("127.0.0.1", 0), web.Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            port = srv.server_address[1]

            def req(method, path, form=None, cookie=""):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                body = urllib.parse.urlencode(form or {})
                c.request(method, path, body if form is not None else None,
                          {"Content-Type": "application/x-www-form-urlencoded", "Cookie": cookie})
                r = c.getresponse()
                return r.status, r.getheader("Set-Cookie") or "", r.read().decode()

            status, _, body = req("GET", "/")
            self.assertIn("配对码", body)
            self.assertEqual(req("POST", "/cred", {"device": "e0:0a", "secret": "x"})[0], 401)
            code = pairing.new_code(Path(d) / "pair.json")
            status, cookie, _ = req("POST", "/pair", {"code": code})
            self.assertEqual(status, 303)
            sid = cookie.split(";")[0]
            req("POST", "/cred", {"device": "e0:0a", "service": "ssh", "username": "pi", "secret": "pw-123"}, sid)
            self.assertEqual(st.vault.get("e0:0a", "ssh").secret, "pw-123")
            _, _, page = req("GET", "/", cookie=sid)
            self.assertIn("SSH 登录 ✓", page)
            self.assertNotIn("pw-123", page)
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
