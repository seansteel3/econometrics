#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import subprocess
import sys
import threading
import time
import webbrowser
import requests

GITHUB_REPO_URL = "https://github.com/seansteel3/econometrics.git"
CLONE_DIR = os.path.expanduser("~/.econ_dash_cache")  # temp local copy


def clone_or_pull_repo():
    if not os.path.exists(CLONE_DIR):
        print("📥 Cloning dashboard repo...")
        subprocess.run(["git", "clone", GITHUB_REPO_URL, CLONE_DIR], check=True)
    else:
        print("🔄 Pulling latest dashboard updates...")
        subprocess.run(["git", "-C", CLONE_DIR, "pull"], check=True)


def open_browser():
    time.sleep(20)  
    webbrowser.open("http://127.0.0.1:8050/")
    
def wait_until_server_is_up(url="http://127.0.0.1:8050/", timeout=360):
    """Polls the server until it responds or times out."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                webbrowser.open(url)
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    print("⚠️ Warning: Dash server did not start within timeout.")

def main():
    parser = argparse.ArgumentParser(
        description="Launch Economic Dashboards"
    )

    parser.add_argument(
        "--report", "-r",
        choices=["full" #, "gas", "oil", "summary"
                 ],
        help="Choose which dashboard to run",
        default="full"
    )
    
    args = parser.parse_args()
    
    sys.path.insert(0, CLONE_DIR)
    
    # Dynamically include the downloaded repo in the path
    sys.path.insert(0, CLONE_DIR)

    # Dynamically import the module after download
    module_name = f"dashboards.{args.report}_dash"
    dashboard_module = importlib.import_module(module_name)
    
    dash_thread = threading.Thread(target=dashboard_module.main)
    dash_thread.start()
    wait_until_server_is_up()

if __name__ == "__main__":
    main()

