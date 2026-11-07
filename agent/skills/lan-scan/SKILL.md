---
name: lan-scan
description: 扫描本地局域网，找出在线的设备（IP、MAC、主机名），也能查某个 IP 在不在线
risk: safe
tags: 扫描, 局域网, 网络, 设备, 在线, ip, 网段, lan, scan
---
用于「扫描本地网络」「局域网里有哪些设备」「10.0.0.15 在不在线」这类请求。只读，不改任何设置。

做法：
1. 直接运行 `bash /userdata/agent/skills/lan-scan/scripts/scan.sh`：自动找到板子所在的网段（默认 /24），并发 ping 一遍，再读 ARP 表，输出每台在线设备的 IP、MAC、主机名（能解析到的话）。
2. 只想查一个地址：`bash /userdata/agent/skills/lan-scan/scripts/scan.sh 10.0.0.15`。
3. 指定网段前缀：`bash /userdata/agent/skills/lan-scan/scripts/scan.sh 192.168.100`。

注意：
- 有些设备（Windows 默认防火墙、手机省电）不回 ping，但 ARP 表里可能还有它，脚本会一并列出并标「不回 ping」。
- 板子有两个地址（10.0.0.176 / 10.0.0.177）属于同一台机器。
- 向主人汇报时说在线台数，再点出主人可能关心的几台（路由器、服务器、电脑）。
