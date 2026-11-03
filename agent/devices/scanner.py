"""Find the devices on the home LAN and guess what they are (stdlib only, no nmap).

A host counts as up when any probed TCP port answers - an open port or a refused connection both
prove someone is there - or when it shows up in the kernel ARP table afterwards. What it is comes from
the open ports, mDNS (avahi-browse), SSDP/UPnP descriptions and web page titles.
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field

# port -> what it suggests
PORTS = {
    22: "ssh", 23: "telnet", 53: "dns", 80: "http", 135: "msrpc", 139: "netbios", 443: "https",
    445: "smb", 548: "afp", 554: "rtsp", 631: "ipp", 1400: "sonos", 1883: "mqtt", 3389: "rdp",
    5000: "nas-http", 5001: "nas-https", 5555: "adb", 6668: "tuya", 8000: "http-alt", 8008: "cast",
    8009: "cast", 8080: "http-alt", 8123: "home-assistant", 8443: "https-alt", 9100: "printer",
    49152: "upnp", 62078: "iphone-sync",
}

KIND_NAMES = {
    "router": "路由器", "windows-pc": "Windows 电脑", "mac": "苹果电脑", "linux": "Linux 主机",
    "nas": "NAS", "printer": "打印机", "camera": "摄像头", "tv": "电视或投屏设备", "speaker": "音箱",
    "phone": "手机", "apple-hub": "Apple TV 或 HomePod", "ap": "无线路由或中继", "home-assistant": "Home Assistant", "smart-home": "智能家居设备",
    "android": "安卓设备", "unknown": "没认出来的设备",
}


@dataclass
class Device:
    ip: str
    mac: str = ""
    hostname: str = ""
    ports: list[int] = field(default_factory=list)
    services: list[str] = field(default_factory=list)  # mDNS service types / SSDP device types
    names: list[str] = field(default_factory=list)  # friendly names from mDNS / SSDP / web titles
    vendor: str = ""  # manufacturer from UPnP, when it says
    kind: str = "unknown"

    @property
    def key(self) -> str:
        return self.mac or self.ip

    @property
    def label(self) -> str:
        # skip machine ids such as "iTunes_Ctrl_D3A60A798CEB4846" or "70B3D5A1B2C3@Living Room"
        readable = [n for n in self.names if not re.match(r"itunes_ctrl_|[0-9a-f]{12}@", n, re.I)]
        return (readable[0] if readable else "") or self.hostname.removesuffix(".lan") or self.ip


def local_network(iface: str | None = None) -> tuple[ipaddress.IPv4Network, set[str]]:
    """The LAN of the default route, and this board's own addresses on it."""
    out = subprocess.run(["ip", "-4", "-o", "route", "show", "default"], capture_output=True, text=True).stdout
    m = re.search(r"\bdev (\S+)", out)
    iface = iface or (m.group(1) if m else "eth0")
    out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", iface], capture_output=True, text=True).stdout
    addrs = re.findall(r"inet (\d+\.\d+\.\d+\.\d+/\d+)", out)
    if not addrs:
        raise RuntimeError(f"no IPv4 address on {iface}")
    net = ipaddress.ip_interface(addrs[0]).network
    if net.prefixlen < 22:  # never sweep a huge range; keep to the board's /24
        net = ipaddress.ip_interface(addrs[0].split("/")[0] + "/24").network
    return net, {a.split("/")[0] for a in addrs}


async def _probe_port(ip: str, port: int, sem: asyncio.Semaphore, timeout: float) -> tuple[bool, bool]:
    """(host is up, port is open)"""
    async with sem:
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout)
            w.close()
            return True, True
        except ConnectionRefusedError:
            return True, False
        except (OSError, asyncio.TimeoutError):
            return False, False


def read_arp() -> dict[str, str]:
    """ip -> mac for complete entries of the kernel ARP table."""
    table = {}
    try:
        with open("/proc/net/arp", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) >= 4 and parts[2] == "0x2" and parts[3] != "00:00:00:00:00:00":
                    table[parts[0]] = parts[3].lower()
    except OSError:
        pass
    return table


def mdns_browse(timeout: float = 4.0) -> dict[str, dict]:
    """ip -> {"names": [...], "services": [...]} from avahi (if installed)."""
    found: dict[str, dict] = {}
    try:
        raw = subprocess.run(["avahi-browse", "-aprt"], capture_output=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or b""
    except OSError:
        return found
    out = raw.decode("utf-8", "surrogateescape")  # device names are not always valid UTF-8
    for line in out.splitlines():
        p = line.split(";")
        if len(p) < 9 or p[0] != "=" or p[2] != "IPv4":
            continue
        # avahi escapes bytes as \ddd (decimal) and the name itself is UTF-8 ("卧室" arrives as \229\141\167...)
        name = re.sub(rb"\\(\d{3})", lambda m: bytes([int(m.group(1))]), p[3].encode("utf-8", "surrogateescape"))
        name = name.decode("utf-8", "replace")
        entry = found.setdefault(p[7], {"names": [], "services": [], "host": p[6]})
        if name not in entry["names"]:
            entry["names"].append(name)
        if p[4] not in entry["services"]:
            entry["services"].append(p[4])
    return found


def ssdp_discover(timeout: float = 2.5) -> dict[str, dict]:
    """ip -> {"names", "services", "vendor"} from UPnP devices answering an M-SEARCH."""
    msg = ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\n"
           "MX: 2\r\nST: ssdp:all\r\n\r\n").encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(0.3)
    locations: dict[str, set[str]] = {}
    try:
        sock.sendto(msg, ("239.255.255.250", 1900))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                data, (ip, _) = sock.recvfrom(4096)
            except socket.timeout:
                continue
            m = re.search(rb"(?im)^location:\s*(\S+)", data)
            if m:
                locations.setdefault(ip, set()).add(m.group(1).decode("latin-1"))
    except OSError:
        pass
    finally:
        sock.close()
    found: dict[str, dict] = {}
    for ip, urls in locations.items():
        entry = found.setdefault(ip, {"names": [], "services": [], "vendor": ""})
        for url in list(urls)[:3]:
            try:
                xml = urllib.request.urlopen(url, timeout=1.5).read(65536).decode("utf-8", "replace")
            except Exception:
                continue
            for tag, dest in (("friendlyName", "names"), ("deviceType", "services")):
                for v in re.findall(rf"<{tag}>([^<]+)</{tag}>", xml):
                    if v not in entry[dest]:
                        entry[dest].append(v.strip())
            m = re.search(r"<manufacturer>([^<]+)</manufacturer>", xml)
            if m and not entry["vendor"]:
                entry["vendor"] = m.group(1).strip()
    return found


async def _http_title(ip: str, port: int, timeout: float = 1.5) -> str:
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout)
        w.write(f"GET / HTTP/1.0\r\nHost: {ip}\r\nUser-Agent: monday-scan\r\n\r\n".encode())
        await w.drain()
        data = await asyncio.wait_for(r.read(8192), timeout)
        w.close()
    except (OSError, asyncio.TimeoutError):
        return ""
    m = re.search(rb"(?is)<title[^>]*>(.*?)</title>", data)
    if not m:
        return ""
    raw = m.group(1).strip()[:80]
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return ""


def _reverse_dns(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except OSError:
        return ""


def classify(d: Device, gateway: str = "") -> str:
    ports, text = set(d.ports), " ".join(d.names + d.services + [d.hostname, d.vendor]).lower()
    if d.ip == gateway:
        return "router"
    if re.search(r"tenda|tp-?link|mercury|水星|netgear|h3c|ruijie|锐捷|openwrt", text):
        return "ap"
    apple = bool(re.search(r"itunes|airtunes|_airplay|_raop|_companion-link|_meshcop|_trel", text)) or 62078 in ports
    if apple and (re.search(r"_meshcop|_trel|homepod|apple ?tv", text) or (62078 in ports and 53 in ports)):
        return "apple-hub"  # Thread border router / home hub: HomePod or Apple TV
    if apple and (22 in ports or 548 in ports or re.search(r"macbook|imac|mac ?mini", text)):
        return "mac"
    if 8123 in ports or "home assistant" in text or "_home-assistant" in text:
        return "home-assistant"
    if 9100 in ports or 631 in ports or "printer" in text or "_ipp" in text:
        return "printer"
    if 554 in ports or "camera" in text or "ipc" in text or "nvr" in text:
        return "camera"
    if {8008, 8009} & ports or "_googlecast" in text or "mediarenderer" in text or "_airplay" in text \
            or re.search(r"\btv\b|电视|小米盒子|投屏", text):
        return "tv"
    if 1400 in ports or "sonos" in text or "speaker" in text or "音箱" in text:
        return "speaker"
    if (not apple and {5000, 5001} & ports) or "synology" in text or "qnap" in text or "nas" in text:
        return "nas"
    if {3389, 135} & ports or (445 in ports and 22 not in ports and 548 not in ports):
        return "windows-pc"
    if 548 in ports or "_afpovertcp" in text or "macbook" in text or "imac" in text:
        return "mac"
    if 62078 in ports or "iphone" in text or "ipad" in text:
        return "phone"
    if 5555 in ports or "android" in text:
        return "android"
    if {6668, 1883} & ports or "tuya" in text or "xiaomi" in text or "yeelight" in text or "_miio" in text:
        return "smart-home"
    if 22 in ports:
        return "linux"
    if d.mac and int(d.mac.split(":")[0], 16) & 0x02 and not ports:
        return "phone"  # randomized (locally administered) MAC, nothing listening: almost always a phone
    return "unknown"


async def scan(*, timeout: float = 0.8, concurrency: int = 384, progress=None) -> list[Device]:
    net, own = local_network()
    gateway = ""
    m = re.search(r"default via (\S+)", subprocess.run(["ip", "-4", "route"], capture_output=True, text=True).stdout)
    if m:
        gateway = m.group(1)
    hosts = [str(h) for h in net.hosts() if str(h) not in own]
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    mdns_f = loop.run_in_executor(None, mdns_browse)
    ssdp_f = loop.run_in_executor(None, ssdp_discover)

    async def probe_host(ip: str) -> tuple[str, bool, list[int]]:
        res = await asyncio.gather(*(_probe_port(ip, p, sem, timeout) for p in PORTS))
        return ip, any(up for up, _ in res), [p for p, (_, op) in zip(PORTS, res) if op]

    results = await asyncio.gather(*(probe_host(ip) for ip in hosts))
    if progress:
        progress("ports done")
    arp = read_arp()
    mdns, ssdp = await mdns_f, await ssdp_f
    devices = {}
    for ip, up, ports in results:
        if up or ip in arp or ip in mdns or ip in ssdp:
            devices[ip] = Device(ip=ip, mac=arp.get(ip, ""), ports=ports)
    for ip, info in list(mdns.items()) + list(ssdp.items()):
        if ip not in devices and ip in hosts:
            devices[ip] = Device(ip=ip, mac=arp.get(ip, ""))
        if ip in devices:
            d = devices[ip]
            d.names += [n for n in info.get("names", []) if n not in d.names]
            d.services += [s for s in info.get("services", []) if s not in d.services]
            d.vendor = d.vendor or info.get("vendor", "")
            if not d.hostname and info.get("host"):
                d.hostname = info["host"]

    async def enrich(d: Device) -> None:
        if not d.hostname:
            try:
                d.hostname = await asyncio.wait_for(loop.run_in_executor(None, _reverse_dns, d.ip), 2.0)
            except asyncio.TimeoutError:
                pass
        for port in (80, 8080):
            if port in d.ports and not d.names:
                title = await _http_title(d.ip, port)
                if title:
                    d.names.append(title)
        d.kind = classify(d, gateway)

    await asyncio.gather(*(enrich(d) for d in devices.values()))
    return sorted(devices.values(), key=lambda d: ipaddress.ip_address(d.ip))
