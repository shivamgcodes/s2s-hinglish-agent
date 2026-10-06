"""CPU self-test of the shared packages (stdlib only; no Needle engine, no torch, no network). D-SINGLE-SOURCE 2026-10-07.

    python3 packages/selftest.py        (run by deploy/run_all_cpu_tests.sh as suite packages_selftest)

Checks: the router package imports from its one location without the engine and without hindi_share / the vendor word
lists; numconv + romanise via router_v2.prepare_transcript; build_data's lazy hindi_share works when asked; the
role-prompt format is the one deploy/common/session.py uses; both pip packages import and build as wheels (D-LEAN-HF).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


def main():
    os.environ.pop("N1_DIR", None)
    sys.path.insert(0, str(PKG / "needle_router" / "v2"))
    import router_v2, n1path, numconv, build_data, resolver_v2, tools_v2  # noqa: E401,F401
    check("n1path -> packages/needle_router/n1", Path(n1path.N1_DIR).resolve() == (PKG / "needle_router" / "n1").resolve(), n1path.N1_DIR)
    check("numconv from packages/needle_router/numconv", Path(numconv.__file__).resolve().parent.name == "numconv", numconv.__file__)
    check("engine not imported", "needle" not in sys.modules or not hasattr(sys.modules["needle"], "__file__") or
          "needle_router" not in str(getattr(sys.modules["needle"], "__file__", "")), sys.modules.get("needle"))
    check("hindi_share not imported by the runtime modules", "hindi_share" not in sys.modules)
    t = router_v2.prepare_transcript("मेरा नंबर नौ आठ सात छह है, order cancel करना है")
    check("prepare_transcript: numconv + romanise", "9876" in t and "cancel" in t and not any("ऀ" <= c <= "ॿ" for c in t), t)
    rec = json.loads((REPO / "deploy" / "common" / "data" / "records_v4.json").read_text())["food_01"]
    s = build_data.system_text(rec)
    check("build_data.system_text (pinned date + record)", s.startswith(build_data.PINNED_DATE) and rec["phone"] in s, s[:80])
    check("hindi_share still not imported after system_text", "hindi_share" not in sys.modules)
    check("7 agent types in tools_v2", len(tools_v2.AGENT_TYPES) == 7, tools_v2.AGENT_TYPES)
    check("lazy hindi_share loads from packages/hinglish_text on use", "gurgaon" in build_data.CITIES and
          Path(sys.modules["hindi_share"].__file__).resolve().parent == (PKG / "hinglish_text").resolve())

    sys.path.insert(0, str(REPO / "deploy" / "common"))
    import session
    import role_prompt
    check("session uses packages/personaplex_lora/role_prompt.py",
          Path(role_prompt.__file__).resolve() == (PKG / "personaplex_lora" / "role_prompt.py").resolve() and
          session.build_role_prompt is role_prompt.build_role_prompt, role_prompt.__file__)
    c = session.session_config("food_01", "g2")
    check("session_config voice/prompt", c["voice"] == role_prompt.voice_for("g2") == "NATM1.pt" and c["role_prompt"].isascii())

    # pip packages (D-LEAN-HF): import as installed packages from this tree, and build the wheels (no network)
    sys.path.insert(0, str(PKG))
    import needle_router
    import personaplex_lora
    check("needle_router package exposes route/resolve_calls", callable(needle_router.route) and
          needle_router.router_v2 is router_v2 and len(needle_router.AGENT_TYPES) == 7)
    check("personaplex_lora.build_role_prompt == role_prompt.build_role_prompt",
          personaplex_lora.build_role_prompt(rec, "g3") == role_prompt.build_role_prompt(rec, "g3"))
    check("personaplex_lora import does not pull torch", "torch" not in sys.modules)
    try:
        import setuptools  # noqa: F401
        have_st = True
    except ImportError:
        have_st = False
    if have_st and os.environ.get("PKG_SELFTEST_WHEELS", "1") == "1":
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            for pkg, must in (("needle_router", ["needle_router/v2/router_v2.py", "needle_router/n1/romanise_lexicon.json",
                                                 "needle_router/n1/tools_json/food_delivery_support.json",
                                                 "needle_router/numconv/numconv.py", "needle_router/__init__.py"]),
                              ("personaplex_lora", ["personaplex_lora/infer/lora_merge.py", "personaplex_lora/role_prompt.py",
                                                    "personaplex_lora/trainer/merge_lora.py",
                                                    "personaplex_lora/trainer/voice_codes/NATF2.columns.pt"])):
                src = Path(d) / "src" / pkg      # build from a copy: setuptools writes build/ + *.egg-info next to the source
                shutil.copytree(PKG / pkg, src, ignore=shutil.ignore_patterns("__pycache__", "build", "*.egg-info"))
                r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q",
                                    "-w", d, str(src)], capture_output=True, text=True)
                whl = sorted(Path(d).glob(pkg.replace("_", "_") + "-*.whl")) or sorted(Path(d).glob("*.whl"))
                names = zipfile.ZipFile(whl[-1]).namelist() if r.returncode == 0 and whl else []
                missing = [m for m in must if m not in names]
                check(f"wheel {pkg} builds with its code + data", r.returncode == 0 and not missing,
                      (r.stderr[-300:], missing))
                for w in Path(d).glob("*.whl"):
                    w.unlink()
    else:
        print("SKIP wheel build (no setuptools in this python)")
    print(f"packages selftest: {'ALL OK' if not FAILS else str(len(FAILS)) + ' FAIL'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
