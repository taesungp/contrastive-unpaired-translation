from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="./datasets/grumpifycat",
                name="grumpifycat_CUT",
                CUT_mode="CUT"
            ),

            # Command 1
            Options(
                dataroot="./datasets/grumpifycat",
                name="grumpifycat_FastCUT",
                CUT_mode="FastCUT",
            )
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        # RussianBlue -> Grumpy Cats dataset does not have test split.
        # Therefore, let's set the test split to be the "train" set.
        return ["python test.py " + str(opt.set(phase='train')) for opt in self.common_options()]
