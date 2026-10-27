#!/usr/bin/env python3
"""Connect to a local Codex app server for explicit session coordination.

Read-only listing and an explicit active-turn message are the only operations.
The caller supplies the Unix socket; no server is started or restarted.
"""

import argparse
import base64
import json
import os
import socket
import struct
import sys


class Client:
    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(15)
        self.sock.connect(path)
        self.stream = self.sock.makefile("rb")
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        self.sock.sendall(
            ("GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
             "Connection: Upgrade\r\nSec-WebSocket-Key: " + key +
             "\r\nSec-WebSocket-Version: 13\r\n\r\n").encode("ascii")
        )
        status = self.stream.readline()
        if not status.startswith(b"HTTP/1.1 101 "):
            raise RuntimeError("app server did not accept WebSocket upgrade")
        while True:
            line = self.stream.readline()
            if line == b"\r\n":
                break
            if not line or len(line) > 8192:
                raise RuntimeError("invalid WebSocket handshake")
        self.next_id = 1
        self.request("initialize", {"clientInfo": {
            "name": "ballotmancy_coordination", "title": "Ballotmancy coordination",
            "version": "1.0"}})
        self.write_json({"method": "initialized", "params": {}})

    def frame(self, opcode, data):
        mask = os.urandom(4)
        n = len(data)
        if n < 126:
            header = bytes((0x80 | opcode, 0x80 | n))
        elif n < 65536:
            header = bytes((0x80 | opcode, 0x80 | 126)) + struct.pack("!H", n)
        else:
            header = bytes((0x80 | opcode, 0x80 | 127)) + struct.pack("!Q", n)
        self.sock.sendall(header + mask + bytes(v ^ mask[i % 4] for i, v in enumerate(data)))

    def write_json(self, value):
        self.frame(1, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def read_exact(self, size):
        data = self.stream.read(size)
        if len(data) != size:
            raise EOFError("app server closed the connection")
        return data

    def read_message(self):
        fragments = bytearray()
        started = False
        while True:
            first, second = self.read_exact(2)
            opcode, final = first & 15, bool(first & 128)
            size = second & 127
            if size == 126:
                size = struct.unpack("!H", self.read_exact(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self.read_exact(8))[0]
            if size > 16 * 1024 * 1024:
                raise RuntimeError("oversized app-server message")
            mask = self.read_exact(4) if second & 128 else None
            payload = self.read_exact(size)
            if mask:
                payload = bytes(v ^ mask[i % 4] for i, v in enumerate(payload))
            if opcode == 9:
                self.frame(10, payload)
                continue
            if opcode == 8:
                raise EOFError("app server closed the WebSocket")
            if opcode == 1 and not started:
                started = True
            elif opcode != 0 or not started:
                raise RuntimeError("unexpected WebSocket frame")
            fragments.extend(payload)
            if len(fragments) > 16 * 1024 * 1024:
                raise RuntimeError("oversized fragmented app-server message")
            if final:
                return json.loads(fragments)

    def request(self, method, params):
        request_id = self.next_id
        self.next_id += 1
        self.write_json({"id": request_id, "method": method, "params": params})
        while True:
            message = self.read_message()
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"]))
                return message["result"]

    def close(self):
        self.stream.close()
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="local app-server Unix socket")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list recent sessions and their runtime status")
    tail = sub.add_parser("recent", help="read a session's recent turn summaries")
    tail.add_argument("thread_id")
    send = sub.add_parser("send", help="send stdin as a message to an active turn")
    send.add_argument("thread_id")
    args = parser.parse_args()
    client = Client(args.socket)
    try:
        if args.action == "list":
            result = client.request("thread/list", {"limit": 50, "sortKey": "updated_at", "useStateDbOnly": True})
            result = [{k: t.get(k) for k in ("id", "name", "status", "cwd", "updatedAt")} for t in result["data"]]
        elif args.action == "recent":
            result = client.request("thread/turns/list", {"threadId": args.thread_id, "limit": 3,
                                    "sortDirection": "desc", "itemsView": "summary"})
        else:
            text = sys.stdin.read()
            if not text.strip():
                raise ValueError("message on stdin must not be empty")
            turns = client.request("thread/turns/list", {"threadId": args.thread_id, "limit": 1,
                                   "sortDirection": "desc", "itemsView": "notLoaded"})["data"]
            if not turns or turns[0]["status"] != "inProgress":
                raise RuntimeError("target has no active turn; no message was sent")
            result = client.request("turn/steer", {"threadId": args.thread_id,
                                    "expectedTurnId": turns[0]["id"],
                                    "input": [{"type": "text", "text": text}]})
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, EOFError, RuntimeError, ValueError) as error:
        print(f"Session coordination failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
