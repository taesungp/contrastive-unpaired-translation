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
    # target_launcher_name = launcher_name.replace('_', '') + 'launcher'
    for name, cls in launcherlib.__dict__.items():
        if name.lower() == "launcher":
            launcher = cls

    if launcher is None:
        raise ValueError("In %s.py, there should be a class named Launcher")

    return launcher


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('cmd')
    parser.add_argument('id', nargs='+', type=str)
    parser.add_argument('--mode', default=None)
    parser.add_argument('--which_epoch', default=None)
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--subdir', default='')
    parser.add_argument('--title', default='')
    parser.add_argument('--gpu_id', default=None, type=int)
    parser.add_argument('--phase', default='test')

    opt = parser.parse_args()

    name = opt.name
    Launcher = find_launcher_using_name(name)

    instance = Launcher()

    cmd = opt.cmd
    ids = 'all' if 'all' in opt.id else [int(i) for i in opt.id]
    if cmd == "launch":
        instance.launch(ids, continue_train=opt.continue_train)
    elif cmd == "stop":
        instance.stop()
    elif cmd == "send":
        assert False
    elif cmd == "close":
        instance.close()
    elif cmd == "dry":
        instance.dry()
    elif cmd == "relaunch":
        instance.close()
        instance.launch(ids, continue_train=opt.continue_train)
    elif cmd == "run" or cmd == "train":
        assert len(ids) == 1, '%s is invalid for run command' % (' '.join(opt.id))
        expid = ids[0]
        instance.run_command(instance.commands(), expid,
                             continue_train=opt.continue_train,
                             gpu_id=opt.gpu_id)
    elif cmd == 'launch_test':
        instance.launch(ids, test=True)
    elif cmd == "run_test" or cmd == "test":
        test_commands = instance.test_commands()
        if ids == "all":
            ids = list(range(len(test_commands)))
        for expid in ids:
            instance.run_command(test_commands, expid, opt.which_epoch,
                                 gpu_id=opt.gpu_id)
            if expid < len(ids) - 1:
                os.system("sleep 5s")
    elif cmd == "print_names":
        instance.print_names(ids, test=False)
    elif cmd == "print_test_names":
        instance.print_names(ids, test=True)
    elif cmd == "create_comparison_html":
        instance.create_comparison_html(name, ids, opt.subdir, opt.title, opt.phase)
    else:
        raise ValueError("Command not recognized")
