# Habitat-Robotics-NavSim
A reinforcement learning environment using Stable-Baselines3.

## Setup
```bash
pip install -r requirements.txt

python main.py 

podman build -t container_name .
podman run -it container_name python main.py


podman ps -a
podman cp CONTAINER_ID:/app ~/personal/backup_dir



tmux new-session -s mysession
Ctrl + b, then d #detaching
tmux attach -t mysession
tmux kill-session -t mysession


Ctrl + b, then c #-new window for tmux session
