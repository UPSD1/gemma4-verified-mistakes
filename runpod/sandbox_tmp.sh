# Where the harness puts subprocess sandboxes (TMPDIR). /dev/shm is mounted noexec on RunPod, so compiled
# extensions (pydantic-core, polars, black) fail to load from it; /workspace is a network filesystem.
# Sandboxes go on the container disk; the harness's unpacked-wheel cache (read as data, ~9 GB) stays in RAM.
export TMPDIR=/root/sbx
mkdir -p "$TMPDIR" /dev/shm/sp_cache
[ -e "$TMPDIR/swegemma_sp_cache_v8" ] || ln -s /dev/shm/sp_cache "$TMPDIR/swegemma_sp_cache_v8"
