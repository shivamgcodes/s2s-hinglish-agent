"""Run a command, print wall seconds and peak child RSS (pod has no /usr/bin/time)."""
import resource, subprocess, sys, time, json
t0 = time.time()
rc = subprocess.call(sys.argv[1:])
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print("TIMED " + json.dumps({"rc": rc, "wall_s": round(time.time() - t0, 1),
      "peak_rss_gb": round(ru.ru_maxrss / 1e6, 2), "user_s": round(ru.ru_utime), "sys_s": round(ru.ru_stime),
      "cmd": " ".join(sys.argv[1:])}), flush=True)
sys.exit(rc)
