# 2026-06-18 Image Rebuild Snapshot

This snapshot records the current container state before rebuilding the image.

Files:
- `newil_environment.yml`: full conda export for the New-IL main environment.
- `newil_conda_explicit.txt`: explicit conda package list for exact reconstruction where possible.
- `newil_pip_freeze.txt`: pip packages inside `newil`.
- `pi_*`: OpenPI environment snapshot, if the `pi` conda env exists.
- `egl_nvidia_apt_packages.tsv`: installed system EGL/GLVND/Mesa/NVIDIA package subset.
- `dpkg_all.tsv`: full dpkg package list.
- `nvidia_egl_userspace_files.tsv`: manually downloaded/extracted NVIDIA EGL userspace files under `/data/L202500340/data/nvidia-egl-580`.
- `zshrc_newil_env.txt`: effective New-IL/cache/EGL environment variables from a fresh zsh shell.

Important EGL note:
The current container has userspace EGL files prepared, but `eglQueryDevicesEXT()` still returned 0. For the rebuilt image, preinstall GLVND/EGL userspace packages and ensure the runtime exposes NVIDIA graphics/display capability if possible.
