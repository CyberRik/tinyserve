# Demo 1: token streaming. Needs `.\scripts\demo.ps1 page` running.
# Render from the repo root:  vhs scripts/demo-1.tape
#
# Shell is cmd, not bash: on Windows, "bash" resolves to System32\bash.exe, the
# WSL launcher, and WSL2's 127.0.0.1 is not the Windows host the server is on.
# cmd has no curl alias (PowerShell's curl is Invoke-WebRequest), and reading the
# body from a file avoids cmd's awkward JSON quoting while keeping it on camera.
Output docs/media/demo-1-streaming.gif
Set Shell "cmd"
Set FontSize 22
Set Width 1200
Set Height 500
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms

# Off camera: wait for the shell, reduce the prompt to "> ", clear.
Hide
Sleep 1s
Type "cd scripts & prompt $G$S & cls"
Enter
Sleep 500ms
Show

Type "type haiku.json"
Enter
Sleep 1500ms
Type `curl -N http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d @haiku.json`
Enter
Sleep 8s
