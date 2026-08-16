"""One-shot upload helper: commit the working tree to GitHub via `gh api`
(used when git push transport is unavailable)."""

import base64
import json
import os
import subprocess
import sys

REPO = "xiaolibai-sys/ComfyUI-MiniMaxH3"
BRANCH = "main"
MESSAGE = """v1.4.0: rolling sampling, audio loudness matching, UI overhaul

- Restructure sampling into the rolling FL2VA pipeline
  (session/runner/memory, batched keyframe encoding, fewer model reloads)
- Add segment audio loudness matching: BS.1770 gated LUFS measurement,
  true-peak limiting, leveler-style gain smoothing, boundary crossfades
- Rework FL Constraint frontend: segment timeline UI, click-to-replace
  images, period delete semantics, in-panel toggles over fl_data
- Fix FP8 video VAE compatibility
- Expand test coverage (rolling pipeline, loudness, frontend UI smoke)
"""


def gh_api(path, method="GET", payload=None):
    cmd = ["gh", "api", f"repos/{REPO}/{path}", "-X", method]
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        cmd += ["--input", "-"]
    r = subprocess.run(cmd, input=data, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh api {method} {path} failed:\n"
                           + r.stderr.decode(errors="replace"))
    return json.loads(r.stdout.decode())


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True).stdout
    uploads, deletes = [], []
    for line in status.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:]
        if "->" in path:  # rename entry: take the new path
            path = path.split("->")[-1].strip()
        if code.strip() == "D":
            deletes.append(path)
        else:
            uploads.append(path)
    print(f"uploading {len(uploads)} files, deleting {len(deletes)}")

    ref = gh_api(f"git/refs/heads/{BRANCH}")
    head_sha = ref["object"]["sha"]
    base_tree = gh_api(f"git/commits/{head_sha}")["tree"]["sha"]
    print(f"base commit {head_sha[:8]}")

    entries = []
    for i, path in enumerate(uploads, 1):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        blob = gh_api("git/blobs", "POST",
                      {"content": b64, "encoding": "base64"})
        entries.append({"path": path.replace(os.sep, "/"),
                        "mode": "100644", "type": "blob",
                        "sha": blob["sha"]})
        print(f"  [{i}/{len(uploads)}] {path}")
    for path in deletes:
        entries.append({"path": path.replace(os.sep, "/"),
                        "mode": "100644", "type": "blob", "sha": None})
        print(f"  delete {path}")

    tree = gh_api("git/trees", "POST",
                  {"base_tree": base_tree, "tree": entries})
    commit = gh_api("git/commits", "POST", {
        "message": MESSAGE,
        "tree": tree["sha"],
        "parents": [head_sha],
    })
    gh_api(f"git/refs/heads/{BRANCH}", "PATCH",
           {"sha": commit["sha"], "force": False})
    print(f"pushed commit {commit['sha'][:8]} to {BRANCH}")


if __name__ == "__main__":
    sys.exit(main())
