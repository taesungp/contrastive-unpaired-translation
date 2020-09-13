from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            Options(
                name="singleimage_monet_etretat",
                dataroot="./datasets/single_image_monet_etretat",
                model="sincut"
            )
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        return ["python test.py " + str(opt) for opt in self.common_options()]
