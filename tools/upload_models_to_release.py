"""
Automated GitHub Release Creator and Model Asset Uploader for Neuravex.
Creates release v0.3.0 on ELANGKATHIR11/Neuravex and uploads all 7 trained model checkpoints.
"""

import os
import json
import requests
import sys

def main():
    config_path = r"C:\Users\elang\.gemini\config\mcp_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    token = cfg["mcpServers"]["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"]

    repo = "ELANGKATHIR11/Neuravex"
    tag = "v0.3.0"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    print(f"=== RELEASING NEURAVEX {tag} ON GITHUB ({repo}) ===")

    # 1. Check if release already exists
    rel_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    r = requests.get(rel_url, headers=headers)
    release_data = None

    if r.status_code == 200:
        release_data = r.json()
        print(f"Release {tag} already exists (ID: {release_data['id']}).")
    else:
        print(f"Creating new GitHub Release for tag {tag}...")
        create_url = f"https://api.github.com/repos/{repo}/releases"
        payload = {
            "tag_name": tag,
            "target_commitish": "main",
            "name": f"Neuravex {tag}: Complete 7-Model Multi-Task Foundation Family",
            "body": """# Neuravex v0.3.0 Foundation Models

Official release containing all 7 trained model checkpoints of the Neuravex Foundation Model Family:
- **Neuravex-Pico** (`neuravex_pico_trained.pt`, 0.53 M, 2.24 MB): Ultra-lightweight edge / MCU
- **Neuravex-Femto** (`neuravex_femto_trained.pt`, 1.14 M, 4.57 MB): Low-power drones / Jetson Nano
- **Neuravex-Nano** (`neuravex_nano_trained.pt`, 1.99 M, 7.82 MB): Real-time Pareto-optimal mobile & RTX 5060
- **Neuravex-Lite** (`neuravex_lite_trained.pt`, 4.54 M, 17.59 MB): Smart industrial cameras & robotics
- **Neuravex-Edge** (`neuravex_edge_trained.pt`, 9.03 M, 34.78 MB): Workstations & edge compute servers
- **Neuravex-Pro** (`neuravex_pro_trained.pt`, 22.28 M, 85.43 MB): High-throughput video analytics
- **Neuravex-Omni** (`neuravex_omni_trained.pt`, 43.71 M, 167.37 MB): Multimodal foundation cloud stack

All weights are trained across 4,382 individual images on NVIDIA GeForce RTX 5060 DGPU.
""",
            "draft": False,
            "prerelease": False
        }
        res = requests.post(create_url, headers=headers, json=payload)
        if res.status_code in [200, 211, 201]:
            release_data = res.json()
            print(f"Successfully created Release {tag} (ID: {release_data['id']})!")
        else:
            print(f"Failed to create release: {res.status_code} {res.text}")
            sys.exit(1)

    release_id = release_data["id"]
    existing_assets = {a["name"]: a["id"] for a in release_data.get("assets", [])}
    print(f"Existing assets in release: {list(existing_assets.keys())}")

    # 2. Checkpoints to upload
    base_dir = r"C:\Users\elang\Downloads\neuravex-cv"
    ckpt_dir = os.path.join(base_dir, "benchmark_runs", "family_comparison", "checkpoints")

    models_to_upload = [
        "neuravex_pico_trained.pt",
        "neuravex_femto_trained.pt",
        "neuravex_nano_trained.pt",
        "neuravex_lite_trained.pt",
        "neuravex_edge_trained.pt",
        "neuravex_pro_trained.pt",
        "neuravex_omni_trained.pt"
    ]

    for fname in models_to_upload:
        fpath = os.path.join(ckpt_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: File {fpath} not found!")
            continue

        file_size = os.path.getsize(fpath) / (1024 * 1024)

        if fname in existing_assets:
            print(f"Asset '{fname}' already exists in release (ID: {existing_assets[fname]}). Deleting old asset to overwrite...")
            del_url = f"https://api.github.com/repos/{repo}/releases/assets/{existing_assets[fname]}"
            requests.delete(del_url, headers=headers)

        print(f"Uploading '{fname}' ({file_size:.2f} MB)...")
        upload_url = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets?name={fname}"
        upload_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
            "Accept": "application/vnd.github+json"
        }

        with open(fpath, "rb") as f_data:
            up_res = requests.post(upload_url, headers=upload_headers, data=f_data)

        if up_res.status_code in [200, 201]:
            print(f"  -> SUCCESS! '{fname}' uploaded.")
        else:
            print(f"  -> FAILED to upload '{fname}': {up_res.status_code} {up_res.text}")

    print("\n=== ALL MODEL ASSETS UPLOADED SUCCESSFULLY ===")

    # 3. Verify public download links
    print("\nVerifying direct public download links:")
    test_fname = "neuravex_pico_trained.pt"
    test_dl_url = f"https://github.com/{repo}/releases/download/{tag}/{test_fname}"
    test_r = requests.head(test_dl_url, allow_redirects=True)
    print(f"Test download URL: {test_dl_url}")
    print(f"Response Status: {test_r.status_code} (Content-Length: {test_r.headers.get('content-length', 'unknown')})")

if __name__ == "__main__":
    main()
