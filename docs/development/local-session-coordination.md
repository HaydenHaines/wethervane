# Local session coordination

When Codex's session tool is unavailable but the local app server is running,
`scripts/codex-session-bridge.py` can use its documented Unix WebSocket
interface to inspect session status or send an explicitly authorized message.
It does not start or restart an app server, create a thread, start a turn,
change approval settings, or retry a failed message. Use only within the
user's instructions and the active session's permissions; it is not a way to
override an approval requirement in another session.

Find the current app-server socket from the process list, for example with
`ss -xlpn`. The path can change; do not rely on a saved socket name. Then:

```bash
python3 scripts/codex-session-bridge.py --socket /path/to/socket list
python3 scripts/codex-session-bridge.py --socket /path/to/socket recent THREAD_ID
printf '%s\n' 'An explicitly authorized coordination update.' | \
  python3 scripts/codex-session-bridge.py --socket /path/to/socket send THREAD_ID
```

`send` accepts text on standard input and only steers an active turn. It
checks the active turn ID immediately before sending; if the turn changes or
is idle, the call fails instead of creating a new turn. An accepted response
means the server accepted the message, not that its recipient has processed
it. Session status or a later reply must be checked separately.

Protocol reference: [Codex App Server](https://developers.openai.com/codex/app-server).
