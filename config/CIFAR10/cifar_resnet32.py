# Testing configurations
config = {}

training_opt = {}
training_opt['dataset'] = 'CIFAR10_LT'
training_opt['num_classes'] = 10
training_opt['batch_size'] = 128
training_opt['num_workers'] = 4
training_opt['num_epochs'] = 200
training_opt['display_step'] = 100
training_opt['feature_dim'] = 64
training_opt['open_threshold'] = 0.1
training_opt['learning_rate'] = 0.1
# training_opt['cifar_imb_ratio'] = 0.01
training_opt['log_dir'] = './logs/CIFAR10_LT'
training_opt['sampler'] = None#{'def_file': './data/ClassAwareSampler.py', 'num_samples_cls': 4, 'type': 'ClassAwareSampler'}
training_opt['scheduler_params'] = {'step_size':int(training_opt['num_epochs']/3), 'gamma': 0.1} # every 10 epochs decrease lr by 0.1
config['training_opt'] = training_opt

networks = {}
feature_param = {'use_modulatedatt': False, 'use_fc': False, 'dropout': None,
                 'stage1_weights': False, 'dataset': training_opt['dataset']}
feature_optim_param = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 2e-4}
networks['feat_model'] = {'def_file': './models/ResNet32Feature.py',
                          'params': feature_param,
                          'optim_params': feature_optim_param,
                          'fix': False}
classifier_param = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes'],
                    'stage1_weights': False, 'dataset': training_opt['dataset']}
classifier_optim_param = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 2e-4}
networks['classifier'] = {'def_file': './models/DotProductClassifier.py',
                          'params': classifier_param,
                          'optim_params': classifier_optim_param}
config['networks'] = networks

criterions = {}
perf_loss_param = {}
criterions['PerformanceLoss'] = {'def_file': './loss/SoftmaxLoss.py', 'loss_params': perf_loss_param,
                                 'optim_params': None, 'weight': 1.0}
config['criterions'] = criterions

memory = {}
memory['centroids'] = False
memory['init_centroids'] = False
config['memory'] = memory