from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                # NOTE: download the resized (and compressed) val set from
                # http://efrosgans.eecs.berkeley.edu/CUT/datasets/cityscapes_val_for_CUT.tar
                dataroot="datasets/cityscapes/cityscapes_val/",
                direction="BtoA",
                phase="val",
                name="cityscapes_cut_pretrained",
                CUT_mode="CUT",
            ),

            # Command 1
            Options(
                dataroot="./datasets/cityscapes_unaligned/cityscapes/",
                direction="BtoA",
                name="cityscapes_fastcut_pretrained",
                CUT_mode="FastCUT",
            ),

            # Command 2
            Options(
                dataroot="./datasets/horse2zebra/",
                name="horse2zebra_cut_pretrained",
                CUT_mode="CUT"
            ),

            # Command 3
            Options(
                dataroot="./datasets/horse2zebra/",
                name="horse2zebra_fastcut_pretrained",
                CUT_mode="FastCUT",
            ),

            # Command 4
            Options(
                dataroot="./datasets/afhq/cat2dog/",
                name="cat2dog_cut_pretrained",
                CUT_mode="CUT"
            ),

            # Command 5
            Options(
                dataroot="./datasets/afhq/cat2dog/",
                name="cat2dog_fastcut_pretrained",
                CUT_mode="FastCUT",
            ),

            
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        return ["python test.py " + str(opt.set(num_test=500)) for opt in self.common_options()]
