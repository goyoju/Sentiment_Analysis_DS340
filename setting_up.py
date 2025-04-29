import subprocess
import sys

def run_script(script_name):
    try:
        python_executable = sys.executable
        subprocess.run([python_executable, script_name], check=True)
        print(f"{script_name} done")
    except subprocess.CalledProcessError as e:
        print(f"{script_name} error: {e}")

if __name__ == "__main__":
    run_script('model/tfrecord_convert.py') #converting data
    run_script('model/training_setting.py') #setting training sets
    run_script('model/modeling.py') #modeling