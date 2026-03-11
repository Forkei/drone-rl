import sys, os
sys.path.insert(0, r"C:\Users\forke\Documents\Drones\PyBullet1")
os.chdir(r"C:\Users\forke\Documents\Drones\PyBullet1")
exec(open("training/active/train_vision_mujoco.py").read())
main()
