"""Runs INSIDE the project sandbox: 127.0.0.1:PORT -> the egress proxy's Unix socket (learning.egress_proxy).

    python3 projbox_fwd.py /tmp/.egress.sock 3128 /tmp/.fwd_ready
The sandbox has no network namespace of its own beyond loopback, so this is its only way out, and the
proxy on the other end only lets public addresses through. Stdlib only; exits with the sandbox.
"""
import select
import socket
import sys
import threading


def pipe(a, b):
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], 300)
            if not r:
                return
            for s in r:
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        pass
    finally:
        a.close()
        b.close()


def main():
    sock_path, port, ready = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(64)
    open(ready, "w").close()
    while True:
        c, _ = srv.accept()
        u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            u.connect(sock_path)
        except OSError:
            c.close()
            u.close()
            continue
        threading.Thread(target=pipe, args=(c, u), daemon=True).start()


if __name__ == "__main__":
    main()
