#!/bin/bash
# 1) Update and install prerequisites

apt-get update && apt-get install -y \
  ca-certificates curl gnupg lsb-release

# 2) Add Docker’s official GPG key and repository
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/$(. /etc/os-release; echo "$ID") \
  $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

# 3) Install Docker
apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io


systemctl enable docker
systemctl start docker


# e.g., pull your custom image
docker pull saigum/state-experiments:latest

cat << 'EOF' > download_state.py
import os
from huggingface_hub import hf_hub_download

# Repository and cache settings
grepo = "arcinstitute/SE-600M"
cache_dir = os.path.expanduser("hf_models")
# Ensure cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Files to download
filenames = [
    "se600m_epoch4.ckpt",
    "protein-embeddings.pt",
    "config.yaml",
]

# Download and symlink
print("Downloading files and creating symlinks:")
for fn in filenames:
    path = hf_hub_download(
        repo_id=grepo,
        filename=fn,
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f" • {fn} → {path}")
    # Symlink in current directory
    link_name = os.path.join(os.getcwd(), fn)
    try:
        if os.path.islink(link_name) or os.path.exists(link_name):
            os.remove(link_name)
        os.symlink(path, link_name)
        print(f"   symlinked as {link_name}")
    except OSError as e:
        print(f"   failed to symlink {fn}: {e}")
EOF

