"""
Author: your name
Date: 2024-10-28 15:09:39
"""

import subprocess
import time

# third-party packages

# user-defined packages


# uvicornのサーバーをサブプロセスとして起動
def start_uvicorn_server():
    # サーバーをサブプロセスで起動する
    process = subprocess.Popen(
        [
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--workers",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def main():
    try:
        process = start_uvicorn_server()
        print("process started")
        time.sleep(10)
    finally:
        process.terminate()
        process.wait()
        print("embedding server is stopped")


if __name__ == "__main__":
    main()
