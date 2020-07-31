from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):

    # List of training commands
    def commands(self):
        opt = Options()

        # common options for all training sessions defined in this launcher
        opt.set(dataroot="~/datasets/cityscapes/",  # specify --dataroot option here
                model="contrastive_cycle_gan",
                pool_size=0,
                no_dropout="",
                init_type="xavier",
                batch_size=1,
                display_freq=400,
                evaluation_metrics="fid,cityscapes",
                evaluation_freq=10000,
                direction="BtoA",
                use_recommended_options="",
                nce_idt_freq=0.1,
                )

        # Specify individual options here
        commands = [

            # first command.
            # This command can be run using python -m experiments placeholder run 0
            # It will output python train.py [OPTIONS], where OPTIONS are everything defined in the variable opt
            "python train.py " + str(opt.clone().set(
                name="cityscapes_placeholder_noidt",  # name of experiments
                nce_idt=False,
            )),

            # second command.
            # This command can be run using python -m experiments placeholder run 1
            # It removes the option --nce_idt_freq 0.1 that was defined by our common options
            "python train.py " + str(opt.clone().set(
                name="cityscapes_placeholder_singlelayer",
                nce_layers="16",
            ).remove("nce_idt_freq")),


            # third command that performs multigpu training
            # This command can be run using python -m experiments placeholder run 2
            "python train.py " + str(opt.clone().set(
                name="cityscapes_placeholder_multigpu",
                nce_layers="16",
                batch_size=4,
                gpu_ids="0,1",
            )),

        ]

        return commands

    # This is the command used for testing.
    # They can be run using python -m experiments placeholder run_test $i
    def test_commands(self):
        opt = Options()
        opt.set(dataroot="~/datasets/cityscapes_unaligned/cityscapes",
                model="contrastive_cycle_gan",
                no_dropout="",
                init_type="xavier",
                batch_size=1,
                direction="BtoA",
                epoch=40,
                phase='train',
                evaluation_metrics="fid",
                )

        commands = [
            "python test.py " + str(opt.clone().set(
                name="cityscapes_nce",
                nce_layers="0,8,16",
                direction="BtoA",
            )),
        ]

        return commands
