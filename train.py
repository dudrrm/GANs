import time
from options.train_options import TrainOptions

# set CUDA_VISIBLE_DEVICES before import torch
# WHY ????
opt = TrainOptions().parse()

from data.data_loader import CreateDataLoader
from models.define_model import create_model
from util.visualizer import Visualizer

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('# training images = {}'.format(dataset_size))

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch {}, total_steps {})'.format(epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_latest_freq == 0:
        print('saving the model at the end of epoch {}, iters {}'.format(epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()
