"""Download an open MNI152 T1 template, brain mask and tissue priors.

Run once with Python 3.7. These are anatomical reference images only; no
ABIDE diagnoses or subject identifiers are downloaded or consulted.
"""

import hashlib
import json
import os
import urllib.request


BASE = "https://raw.githubusercontent.com/neuroconductor/MNITemplate/master/inst/extdata/"
FILES = (
    "MNI152_T1_2mm.nii.gz",
    "MNI152_T1_2mm_Brain.nii.gz",
    "MNI152_T1_2mm_Brain_Mask.nii.gz",
    "MNI152_T1_2mm_Brain_FAST_pve_0.nii.gz",
    "MNI152_T1_2mm_Brain_FAST_pve_1.nii.gz",
    "MNI152_T1_2mm_Brain_FAST_pve_2.nii.gz",
)


def main():
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
    os.makedirs(target, exist_ok=True)
    provenance = {"source": "https://github.com/neuroconductor/MNITemplate",
                  "files": {}}
    for name in FILES:
        path = os.path.join(target, name)
        if not os.path.exists(path):
            print("Downloading " + name, flush=True)
            with urllib.request.urlopen(BASE + name, timeout=120) as response:
                data = response.read()
            if not data.startswith(b"\x1f\x8b"):
                raise ValueError("Download is not gzip: " + name)
            with open(path, "wb") as output:
                output.write(data)
        with open(path, "rb") as source:
            data = source.read()
        provenance["files"][name] = {"url": BASE + name,
                                     "bytes": len(data),
                                     "sha256": hashlib.sha256(data).hexdigest()}
    with open(os.path.join(target, "provenance.json"), "w", encoding="utf-8") as output:
        json.dump(provenance, output, indent=2)
    print("Downloaded %d reference files." % len(FILES), flush=True)


if __name__ == "__main__":
    main()
