# Workspace Instructions

- When a task reaches a natural phase boundary and the thread is getting large enough that chat reload latency may become annoying, recommend starting a new chat. Good points include: after a debugging pass, after launching or handing off a run, after post-run analysis, after a patch lands and verification finishes, or after a long exchange with many logs, screenshots, or command outputs.
- When recommending a restart, include a short paste-ready handoff prompt with: current goal, relevant paths, active job IDs or tmux sessions, what has already been concluded, and the exact next step.
- Treat restart suggestions as recommendations, not interruptions; do not repeatedly suggest restarting unless the thread is clearly becoming unwieldy.

- For cleanup requests involving FID artifacts or generated images, default to preserving metadata, manifests, summaries, and comparison outputs.
- Never delete FID manifests, manifest summaries, sweep logs, reference `.npz` files, comparison plots, or regenerated evaluation metadata unless the user explicitly asks for those exact files to be removed.
- When the user asks to remove only generated samples used for FID, limit deletion to disposable sample-image dumps such as `samples/*.png`, epoch-by-epoch sample plots, reverse-process image dumps, and similarly obvious generated rasters.
- If it is ambiguous whether a file is disposable output or evaluation or sweep bookkeeping, preserve it by default and ask before deleting.
