---
name: ssh-remote
description: 连接和操作局域网里的其他电脑、服务器（SSH），包括第一次只有账号密码时怎么接入
risk: ask
tags: ssh, 远程, 登录, 服务器, 电脑, 主机, 连接, windows, linux
---
已配置好的机器直接用 ssh 工具，host 写别名（规划提示里的「已配置好密钥登录的主机」列表）。常用别名：
- 196：编译服务器 gp@192.168.100.196（硬盘紧张，剩十几 G）
- backup：备份服务器 ikun@192.168.100.3（备份在 /srv/nextcloud-data/monday-backups）
- gpu：A100 训练服务器 gpu@192.168.100.4（星期一的密钥要主人装好后才能用）

第一次连一台新机器：
1. 先用 lan-scan 确认它在线、22 端口开着（`bash /userdata/agent/skills/lan-scan/scripts/scan.sh <IP>`）。
2. 没有密钥时，请主人说出账号和密码；星期一会用密码登录一次、装上自己的公钥、建好别名，然后丢掉密码（老师看不到密码）。
3. 之后一律用别名。

Windows 电脑：需要先在 Windows 上开启「OpenSSH 服务器」（设置 → 系统 → 可选功能），并允许防火墙 22 端口；
没开的话 22 端口不通，告诉主人怎么开，不要反复重试。命令用 PowerShell 语法（`powershell -Command ...`）。

注意：远程执行会改东西的命令（装软件、删文件、重启服务）都要主人确认；只读命令（df、free、uptime、nvidia-smi）一般确认一次可记住。
