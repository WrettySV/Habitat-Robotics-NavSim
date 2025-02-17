import os
import neptune

def init_neptune(project_name="~/Habitat-Robotics-NavSim"):

    api_token = os.getenv("NEPTUNE_API_TOKEN")  
    if api_token is None:
        raise ValueError("Neptune API token not found! Set NEPTUNE_API_TOKEN env variable.")
    
    run = neptune.init_run(
        project="svetl.luckina2016/Habitat-Robotics-NavSim",
        api_token=api_token,
    ) 
    return run


