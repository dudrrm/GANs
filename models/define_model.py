
def create_model(opt):
    """Create a model which the user define."""
    model = None
    print(opt.model)

    if opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        assert (opt.align_data == True)
        model = Pix2PixModel()

    elif opt.model == 'cycle_gan':
        from .cycle_gan_model import CycleGANModel
        #assert(opt.align_data == False)
        model = CycleGANModel()

    else:
        raise ValueError("Model {} is not recognized.".formet(opt.model))

    model.initialize(opt)
    print("model {} was created.".format(model.name()))
    return model
