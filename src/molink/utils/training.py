import os


def get_last_checkpoint(save_dir: str):
    if not save_dir or not os.path.exists(save_dir):
        return None
    filenames = [f for f in os.listdir(save_dir) if f.endswith(".ckpt")]
    if not filenames:
        return None
    def step_from_name(name: str) -> int:
        base = os.path.splitext(name)[0]
        try:
            return int(base)
        except ValueError:
            return -1
    last_filename = sorted(filenames, key=step_from_name)[-1]
    return os.path.join(save_dir, last_filename)
