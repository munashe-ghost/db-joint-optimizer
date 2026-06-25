import random
import time

def inject_failure():
    while True:
        if random.random() < 0.1:
            print("Simulating failure spike...")
        time.sleep(30)

if __name__ == "__main__":
    inject_failure()
