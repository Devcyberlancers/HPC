import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILES = ["api.py", "disease.py", "root_stress.py"]


def launch(script_name: str):
    script_path = BASE_DIR / script_name
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen([sys.executable, str(script_path)], cwd=str(BASE_DIR), env=env)


def main():
    missing = [name for name in FILES if not (BASE_DIR / name).exists()]
    if missing:
        print("Missing files:", ", ".join(missing))
        sys.exit(1)

    processes = []
    try:
        print("[MAIN] Starting api.py ...")
        api_proc = launch("api.py")
        processes.append(("api.py", api_proc))

        time.sleep(2)

        print("[MAIN] Starting disease.py ...")
        disease_proc = launch("disease.py")
        processes.append(("disease.py", disease_proc))

        print("[MAIN] Starting root_stress.py ...")
        root_proc = launch("root_stress.py")
        processes.append(("root_stress.py", root_proc))

        print("\n[MAIN] All services started.")
        print("[MAIN] API         : http://127.0.0.1:18085/state")
        print("[MAIN] Disease UI  : http://127.0.0.1:8088/")
        print("[MAIN] Root Stress : http://127.0.0.1:8080/")
        print("[MAIN] Press Ctrl+C to stop all services.\n")

        while True:
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(f"{name} stopped unexpectedly with exit code {code}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[MAIN] Stopping all services...")
    except Exception as exc:
        print(f"\n[MAIN] Error: {exc}")
    finally:
        for name, proc in reversed(processes):
            if proc.poll() is None:
                print(f"[MAIN] Terminating {name} ...")
                proc.terminate()

        deadline = time.time() + 5
        for name, proc in reversed(processes):
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    print(f"[MAIN] Killing {name} ...")
                    proc.kill()

        print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
