# -*- coding: utf-8 -*-
import os
import subprocess

def run_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True)
        print(f"{script_path} done")
    except subprocess.CalledProcessError as e:
        print(f"{script_path} error: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'model', 'sentiment_predict.py')
    run_script(script_path)