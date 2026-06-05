import subprocess

# Execute the first program
subprocess.run(["python", "emoji.py"])

# Execute the second program after the first program has finished
subprocess.run(["python", "HandGestureRecognize.py"])