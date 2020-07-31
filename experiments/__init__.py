import os
import importlib


def find_launcher_using_name(launcher_name):
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # pythonfiles = glob.glob(cur_dir + '/**/*.py')
    launcher_filename = "experiments.{}_launcher".format(launcher_name)
    launcherlib = importlib.import_module(launcher_filename)

    # In the file, the class called LauncherNameLauncher() will
    # be instantiated. It has to be a subclass of BaseLauncher,
    # and it is case-insensitive.
    launcher = None
    target_launcher_name = launcher_name.replace('_', '') + 'launcher'
    for name, cls in launcherlib.__dict__.items():
        if name.lower() == target_launcher_name.lower():
            launcher = cls

    if launcher is None:
        raise ValueError("In %s.py, there should be a subclass of BaseLauncher "
                         "with class name that matches %s in lowercase." %
                         (launcher_filename, target_launcher_name))

    return launcher


if __name__ == "__main__":
    import sys
    import pickle

    assert len(sys.argv) >= 3

    name = sys.argv[1]
    Launcher = find_launcher_using_name(name)

    cache = "/tmp/tmux_launcher/{}".format(name)
    if os.path.isfile(cache):
        instance = pickle.load(open(cache, 'r'))
    else:
        instance = Launcher()

    cmd = sys.argv[2]
    if cmd == "launch":
        instance.launch()
    elif cmd == "stop":
        instance.stop()
    elif cmd == "send":
        expid = int(sys.argv[3])
        cmd = int(sys.argv[4])
        instance.send_command(expid, cmd)

    os.makedirs("/tmp/tmux_launcher/", exist_ok=True)
    pickle.dump(instance, open(cache, 'w'))
