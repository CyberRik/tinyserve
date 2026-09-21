# Demo 3: burst admission control. Needs `.\scripts\demo.ps1 burst` running
# (config B), which also restores benchmarks/results/burst.* when it stops.
# Render from the repo root:  vhs scripts/demo-3.tape
# Shell is cmd for the same reason as demo-1.tape.
Output docs/media/demo-3-burst.gif
Set Shell "cmd"
Set FontSize 22
Set Width 1200
Set Height 400
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms

Hide
Sleep 1s
Type "prompt $G$S & cls"
Enter
Sleep 500ms
Show

Type "uv run python benchmarks/burst.py --sizes 64 --max-tokens 24"
Enter
# Real wall time of the burst; do not shorten it in post.
Wait+Screen@60s /wrote/
Sleep 3s
