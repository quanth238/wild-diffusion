# GM-CDRO Instructions

## Scope

- Rules in this file apply within the `GM-CDRO` project checkout at `/home/bachlc/GM-CDRO`.
- These rules supplement `/home/bachlc/AGENTS.md`.
- For GM-CDRO artifact handling, treat the storage tree at `/mnt/data/bachlc/GM-CDRO` as part of the same project scope.
- When a GM-CDRO-specific rule conflicts with a more general workspace default, follow the GM-CDRO rule for files and artifacts inside GM-CDRO paths.

## Cleanup and Preservation

- For cleanup requests involving FID artifacts or generated images, default to preserving metadata and comparison outputs.
- For disk cleanup under `GM-CDRO`, inspect disk usage first, identify the largest candidates, and separate disposable generated rasters from metadata or result artifacts before deleting anything.
- Before deleting substantial training artifacts, propose the cleanup set first with the affected paths, artifact classes, and expected savings.
- Never delete FID manifests, manifest summaries, sweep logs, reference `.npz` files, comparison plots, or regenerated evaluation metadata unless the user explicitly asks for those exact files to be removed.
- When the user asks to remove only generated samples used for FID, limit deletion to disposable sample-image dumps such as `samples/*.png`, epoch-by-epoch sample plots, reverse-process image dumps, and similarly obvious generated rasters.
- For conservative `GM-CDRO` cleanup, prefer this order: delete generated FID sample rasters first, then `training-state-*.pt` from completed runs, then prune surplus `network-snapshot-*.pkl`.
- When pruning `network-snapshot-*.pkl`, keep every checkpoint path exactly referenced by current `fid-sweeps` manifests, logs, or evaluation outputs. If a run has no exact references, keep the latest snapshot as a hedge unless the user explicitly asks for full removal.
- Recheck live Slurm and tmux state before substantial deletions, and avoid touching paths used by active runs or current `/mnt/data` work.
- Keep curve plots, compare summaries, manifest bookkeeping, and regenerated evaluation metadata unless the user explicitly requests their removal.
- If it is ambiguous whether a file is disposable output or evaluation or sweep bookkeeping, preserve it by default and ask before deleting.
