"""Build an unmodified cpuminer-opt Git revision in a fresh, isolated directory."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

# Pin the language dialect as well as the CPU target for repeatable A/B builds.
CFLAGS = "-O3 -march=native -std=gnu11 -Wall"
APT_PACKAGES = ("git python3 build-essential autoconf automake libssl-dev "
                "libcurl4-openssl-dev libjansson-dev libgmp-dev zlib1g-dev")
# Linux exposes these names only when the guest OS supports the vector context.
CPU_FLAGS = {"avx512f", "avx512bw", "avx512dq", "avx512vl", "aes", "avx2"}
CC_MACROS = {"__AVX512F__", "__AVX512BW__", "__AVX512DQ__", "__AVX512VL__",
             "__AES__", "__AVX2__", "__x86_64__"}


def check_build_host(cc):
    """Validate the actual compiler target before starting a full miner build."""
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("The native benchmark requires x86-64 Linux.")
    needed = ("git", "tar", "sh", "aclocal", "autoheader", "automake", "autoconf",
              "g++", "make", cc)
    missing = [program for program in needed if not shutil.which(program)]
    if missing:
        raise RuntimeError("Missing build commands: " + ", ".join(missing) +
                           ". Debian/Ubuntu packages: " + APT_PACKAGES)
    defines = subprocess.check_output([cc, *CFLAGS.split(), "-dM", "-E", "-x", "c", "-"],
                                      input="", text=True)
    macros = {line.split()[1] for line in defines.splitlines() if line.startswith("#define ")}
    if "__GNUC__" not in macros or "__clang__" in macros:
        raise RuntimeError("Use GCC, e.g. --cc gcc. The unmodified v26.1 SIMD headers "
                           "use an immediate operand that some Clang versions reject.")
    absent = CC_MACROS - macros
    if absent:
        raise RuntimeError("Compiler -march=native lacks " + ", ".join(sorted(absent)) +
                           ". Check the VM CPU flags and GCC version.")


def check_cpu_flags(cpu):
    """Check the Linux flags of the exact logical CPU used for the benchmark."""
    for block in Path("/proc/cpuinfo").read_text().strip().split("\n\n"):
        fields = {key.strip(): value.strip() for key, value in
                  (line.split(":", 1) for line in block.splitlines() if ":" in line)}
        if fields.get("processor") == str(cpu):
            missing = CPU_FLAGS - set(fields.get("flags", "").split())
            if missing:
                raise RuntimeError(f"CPU {cpu} lacks " + ", ".join(sorted(missing)))
            return fields.get("model name", "unknown")
    raise RuntimeError(f"No /proc/cpuinfo record for CPU {cpu}.")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_variants(repo: Path, dest: Path, cc: str, jobs: int) -> dict:
    """Build all four SHA modes of this revision, checking each actual set of objects.

    All files go into a new destination. The checkout and its uncommitted changes
    are not consulted after exporting the local Git object.
    """
    if jobs < 1:
        raise ValueError("jobs must be positive")
    check_build_host(cc)
    repo, dest = Path(repo).resolve(), Path(dest).resolve()
    rev = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--verify",
                                   "HEAD^{commit}"], text=True).strip()
    dest.mkdir(parents=True, exist_ok=False)
    src, bins = dest / "source", dest / "bin"
    src.mkdir()
    bins.mkdir()
    cc_path = shutil.which(cc)
    env = dict(os.environ, CC=cc_path, CXX="g++", CFLAGS=CFLAGS,
               CXXFLAGS="-O3 -march=native -Wall", LC_ALL="C")
    log_file = dest / "build.log"
    meta = {"commit": rev, "compiler": subprocess.check_output([cc_path, "--version"],
            text=True).splitlines()[0], "cflags": CFLAGS, "commands": [], "variants": {}}
    switches = {
        "combined": (),
        "base": ("SHA256D_BENCH_BASE_ROUNDS", "SHA256D_BENCH_BASE_TERNARY"),
        "rounds": ("SHA256D_BENCH_BASE_TERNARY",),
        "ternary": ("SHA256D_BENCH_BASE_ROUNDS",),
    }
    paths = {}
    print(f"Building {rev[:12]} with {jobs} make job(s); log: {log_file}", flush=True)
    with log_file.open("w") as log:
        def run(command):
            meta["commands"].append(command)
            log.write("$ " + shlex.join(command) + "\n")
            log.flush()
            status = subprocess.run(command, cwd=src, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            if status:
                raise RuntimeError(f"{command[0]} exited with {status}. See {log_file}. "
                                   f"Debian/Ubuntu build packages: {APT_PACKAGES}")
        try:
            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                archive = str(Path(tmp) / "revision.tar")
                run(["git", "-C", str(repo), "archive", "--format=tar",
                     "--output=" + archive, rev])
                run(["tar", "-xf", archive, "-C", str(src)])
            source = src / "algo/sha/sha256-hash-4way.c"
            original = source.read_bytes()
            for define in switches["base"]:
                if define.encode() not in original:
                    raise RuntimeError(f"Revision {rev[:12]} lacks the {define} build switch.")
            run(["sh", "autogen.sh"])
            run(["./configure", "--with-curl=/usr"])
            verify_objects = [
                "algo/sha/cpuminer-sha256-hash-4way.o", "algo/sha/cpuminer-sha256-hash.o",
                "algo/sha/cpuminer-sha256d.o", "algo/sha/cpuminer-sha256d-4way.o",
                "simd-utils/cpuminer-simd-constants.o",
            ]
            sha_obj = src / verify_objects[0]
            for label, defines in switches.items():
                print(f"Building and checking {label} ...", flush=True)
                if defines:
                    prefix = "".join(f"#define {define} 1\n" for define in defines)
                    source.write_bytes(prefix.encode() + original)
                    # Force exactly this object to be recompiled even on coarse-mtime filesystems.
                    sha_obj.unlink()
                    (src / "cpuminer").unlink()
                run(["make", "-j" + str(jobs)])
                run([cc_path, *CFLAGS.split(), "-fno-strict-aliasing", "-I.", "-Icompat/jansson",
                     "bench/sha256d/verify.c", *verify_objects, "-lcrypto", "-o", "verify-sha256d"])
                run(["./verify-sha256d"])
                binary = bins / label
                shutil.copy2(src / "cpuminer", binary)
                paths[label] = binary
                meta["variants"][label] = {"defines": list(defines),
                    "binary_sha256": _sha256(binary), "sha_object_sha256": _sha256(sha_obj)}
        finally:
            (dest / "build.json").write_text(json.dumps(meta, indent=2) + "\n")
    return paths
