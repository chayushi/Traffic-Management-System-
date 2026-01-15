
def safe_close_env(env):
    try:
        if traci.isLoaded():
            traci.close()
    except Exception as e:
