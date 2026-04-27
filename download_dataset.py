import os
import subprocess

DATASETS = {
    "edges2shoes":    "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz",
    "edges2handbags": "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz",
    "facades":        "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz",
    "maps":           "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz",
}


def download(name: str, dest: str = "./data/raw"):
    assert name in DATASETS, f"Unknown dataset. Choose from: {list(DATASETS)}"
    os.makedirs(dest, exist_ok=True)
    url = DATASETS[name]
    fname = os.path.join(dest, os.path.basename(url))
    print(f"Downloading {name} ...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", fname, url], check=True)
    print("Extracting ...")
    subprocess.run(["tar", "-xzf", fname, "-C", dest], check=True)
    os.remove(fname)
    print(f"Done → {dest}/{name}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="edges2shoes", choices=DATASETS)
    p.add_argument("--dest",    default="./data/raw")
    args = p.parse_args()
    download(args.dataset, args.dest)
