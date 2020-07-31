"""
experiment launcher using tmux panes
"""
import os
import math
import GPUtil
import re

available_gpu_devices = None


class Options():
    def __init__(self, *args, **kwargs):
        self.args = []
        self.kvs = {"gpu_ids": "0"}
        self.set(*args, **kwargs)

    def set(self, *args, **kwargs):
        for a in args:
            self.args.append(a)
        for k, v in kwargs.items():
            self.kvs[k] = v

        return self

    def remove(self, *args):
        for a in args:
            if a in self.args:
                self.args.remove(a)
            if a in self.kvs:
                del self.kvs[a]

        return self

    def update(self, opt):
        self.args += opt.args
        self.kvs.update(opt.kvs)
        return self

    def __str__(self):
        final = " ".join(self.args)
        for k, v in self.kvs.items():
            final += " --{} {}".format(k, v)

        return final

    def clone(self):
        opt = Options()
        opt.args = self.args.copy()
        opt.kvs = self.kvs.copy()
        return opt


def grab_pattern(pattern, text):
    found = re.search(pattern, text)
    if found is not None:
        return found[1]
    else:
        None


# http://code.activestate.com/recipes/252177-find-the-common-beginning-in-a-list-of-strings/
def findcommonstart(strlist):
    prefix_len = ([min([x[0] == elem for elem in x])
                   for x in zip(*strlist)] + [0]).index(0)
    prefix_len = max(1, prefix_len - 4)
    return strlist[0][:prefix_len]


class TmuxLauncher():
    def __init__(self):
        super().__init__()
        self.tmux_prepared = False

    def prepare_tmux_panes(self, num_experiments, dry=False):
        self.pane_per_window = 1
        self.n_windows = int(math.ceil(num_experiments / self.pane_per_window))
        print('preparing {} tmux panes'.format(num_experiments))
        for w in range(self.n_windows):
            if dry:
                continue
            window_name = "experiments_{}".format(w)
            os.system("tmux new-window -n {}".format(window_name))
        self.tmux_prepared = True

    def refine_command(self, command, which_epoch, continue_train, gpu_id=None):
        command = str(command)
        if "--gpu_ids" in command:
            gpu_ids = re.search(r'--gpu_ids ([\d,?]+)', command)[1]
        else:
            gpu_ids = "0"

        gpu_ids = gpu_ids.split(",")
        num_gpus = len(gpu_ids)
        global available_gpu_devices
        if available_gpu_devices is None and gpu_id is None:
            available_gpu_devices = [str(g) for g in GPUtil.getAvailable(limit=8, maxMemory=0.5)]
        if gpu_id is not None:
            available_gpu_devices = [i for i in str(gpu_id)]
        if len(available_gpu_devices) < num_gpus:
            raise ValueError("{} GPU(s) required for the command {} is not available".format(num_gpus, command))
        active_devices = ",".join(available_gpu_devices[:num_gpus])
        if which_epoch is not None:
            which_epoch = " --epoch %s " % which_epoch
        else:
            which_epoch = ""
        command = "CUDA_VISIBLE_DEVICES={} {} {}".format(active_devices, command, which_epoch)
        if continue_train:
            command += " --continue_train "

        # available_gpu_devices = [str(g) for g in GPUtil.getAvailable(limit=8, maxMemory=0.8)]
        available_gpu_devices = available_gpu_devices[num_gpus:]

        return command

    def send_command(self, exp_id, command, dry=False, continue_train=False):
        command = self.refine_command(command, None, continue_train=continue_train)
        pane_name = "experiments_{windowid}.{paneid}".format(windowid=exp_id // self.pane_per_window,
                                                             paneid=exp_id % self.pane_per_window)
        if dry is False:
            os.system("tmux send-keys -t {} \"{}\" Enter".format(pane_name, command))

        print("{}: {}".format(pane_name, command))
        return pane_name

    def run_command(self, command, ids, which_epoch=None, continue_train=False, gpu_id=None):
        if type(command) is not list:
            command = [command]
        if ids is None:
            ids = list(range(len(command)))
        if type(ids) is not list:
            ids = [ids]

        for id in ids:
            this_command = command[id]
            refined_command = self.refine_command(this_command, which_epoch, continue_train=continue_train, gpu_id=gpu_id)
            print(refined_command)
            os.system(refined_command)

    def commands(self):
        return []

    def launch(self, ids, test=False, dry=False, continue_train=False):
        commands = self.test_commands() if test else self.commands()
        if type(ids) is list:
            commands = [commands[i] for i in ids]
        if not self.tmux_prepared:
            self.prepare_tmux_panes(len(commands), dry)
            assert self.tmux_prepared

        for i, command in enumerate(commands):
            self.send_command(i, command, dry, continue_train=continue_train)

    def dry(self):
        self.launch(dry=True)

    def stop(self):
        num_experiments = len(self.commands())
        self.pane_per_window = 4
        self.n_windows = int(math.ceil(num_experiments / self.pane_per_window))
        for w in range(self.n_windows):
            window_name = "experiments_{}".format(w)
            for i in range(self.pane_per_window):
                os.system("tmux send-keys -t {window}.{pane} C-c".format(window=window_name, pane=i))

    def close(self):
        num_experiments = len(self.commands())
        self.pane_per_window = 1
        self.n_windows = int(math.ceil(num_experiments / self.pane_per_window))
        for w in range(self.n_windows):
            window_name = "experiments_{}".format(w)
            os.system("tmux kill-window -t {}".format(window_name))

    def print_names(self, ids, test=False):
        if test:
            cmds = self.test_commands()
        else:
            cmds = self.commands()
        if type(ids) is list:
            cmds = [cmds[i] for i in ids]

        for cmdid, cmd in enumerate(cmds):
            name = grab_pattern(r'--name ([^ ]+)', cmd)
            print(name)

    def create_comparison_html(self, expr_name, ids, subdir, title, phase):
        cmds = self.test_commands()
        if type(ids) is list:
            cmds = [cmds[i] for i in ids]

        no_easy_label = True
        dirs = []
        labels = []
        for cmdid, cmd in enumerate(cmds):
            name = grab_pattern(r'--name ([^ ]+)', cmd)
            which_epoch = grab_pattern(r'--epoch ([^ ]+)', cmd)
            if which_epoch is None:
                which_epoch = "latest"
            label = grab_pattern(r'--easy_label "([^"]+)"', cmd)
            if label is None:
                label = name
            else:
                no_easy_label = False
            labels.append(label)
            dir = "results/%s/%s_%s/%s/" % (name, phase, which_epoch, subdir)
            dirs.append(dir)

        commonprefix = findcommonstart(labels) if no_easy_label else ""
        labels = ['"' + label[len(commonprefix):] + '"' for label in labels]
        dirstr = ' '.join(dirs)
        labelstr = ' '.join(labels)

        command = "python ~/tools/html.py --web_dir_prefix results/comparison_ --name %s --dirs %s --labels %s --image_width 256" % (expr_name + '_' + title, dirstr, labelstr)
        print(command)
        os.system(command)
