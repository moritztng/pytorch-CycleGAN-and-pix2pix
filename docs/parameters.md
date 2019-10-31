# Base Options
## Basic
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| dataroot | True | - | path to images (should have subfolders trainA, trainB, valA, valB, etc) |
| name | False | experiment_name | name of the experiment. It decides where to store samples and models |
| gpu_ids | False | 0 | gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU |
| checkpoints_dir | False | ./checkpoints | models are saved here |
## Model
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| model | False | cycle_gan | chooses which model to use. [cycle_gan , pix2pix , test , colorization] |
| input_nc | False | 3 | # of input image channels: 3 for RGB and 1 for grayscale |
| output_nc | False | 3 | # of output image channels: 3 for RGB and 1 for grayscale |
| ngf | False | 64 | # of gen filters in the last conv layer |
| ndf | False | 64 | # of discrim filters in the first conv layer |
| netD | False | basic | specify discriminator architecture [basic , n_layers , pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator |
| netG | False | resnet_9blocks | specify generator architecture [resnet_9blocks , resnet_6blocks , unet_256 , unet_128] |
| n_layers_D | False | 3 | only used if netD==n_layers |
| norm | False | instance | instance normalization or batch normalization [instance , batch , none] |
| init_type | False | normal | network initialization [normal , xavier , kaiming , orthogonal] |
| init_gain | False | 0.02 | scaling factor for normal, xavier and orthogonal. |
| no_dropout | False | - | no dropout for the generator |
## Dataset
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| dataset_mode | False | unaligned | chooses how datasets are loaded. [unaligned , aligned , single , colorization] |
| direction | False | AtoB | AtoB or BtoA |
| serial_batches | False | - | if true, takes images in order to make batches, otherwise takes them randomly |
| num_threads | False | 4 | # threads for loading data |
| batch_size | False | 1 | input batch size |
| load_size | False | 286 | scale images to this size |
| crop_size | False | 256 | then crop to this size |
| max_dataset_size | False | float("inf") | Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded. |
| preprocess | False | resize_and_crop | scaling and cropping of images at load time [resize_and_crop , crop , scale_width , scale_width_and_crop , none] |
| no_flip | False | - | if specified, do not flip the images for data augmentation |
| display_winsize | False | 256 | display window size for both visdom and HTML |
## Additional
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| epoch | False | latest | which epoch to load? set to latest to use latest cached model |
| load_iter | False | 0 | which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch] |
| verbose | False | - | if specified, print more debugging information |
| suffix | False | '' | customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size} |
# Train Options
## visdom and HTML visualization
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| display_freq| False | 400 | frequency of showing training results on screen |
| display_ncols | False | 4 | if positive, display all images in a single visdom web panel with certain number of images per row. |
| display_id | False | 1 | window id of the web display |
| display_server | False | http://localhost | visdom server of the web display |
| display_env | False | main | visdom display environment name (default is "main") |
| display_port | False | 8097 | visdom port of the web display |
| update_html_freq | False | 1000 | frequency of saving training results to html |
| print_freq | False | 100 | frequency of showing training results on console |
| no_html | False |  | do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/ |
## network saving and loading
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| save_latest_freq | False | 5000 | frequency of saving the latest results |
| save_epoch_freq | False | 5 | frequency of saving checkpoints at the end of epochs |
| save_by_iter | False | | whether saves model by iteration |
| continue_train | False | | continue training: load the latest model |
| epoch_count | False | 1 | the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ... |
| phase | False | train | train, val, test, etc |
## training
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| niter | False | 100 | # of iter at starting learning rate |
| niter_decay | False | 100 | # of iter to linearly decay learning rate to zero |
| beta1 | False | 0.5 | momentum term of adam |
| lr | False | 0.0002 | initial learning rate for adam |
| gan_mode | False | lsgan | the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper. |
| pool_size | False | 50 | the size of image buffer that stores previously generated images |
| lr_policy | False | linear | learning rate policy. [linear , step , plateau , cosine] |
| lr_decay_iters | False | 50 | multiply by a gamma every lr_decay_iters iterations |
# Test Options
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| ntest | False | float("inf") | # of test examples. |
| results_dir | False | ./results/ | saves results here. |
| aspect_ratio | False | 1.0 | aspect ratio of result images |
| phase | False | test | train, val, test, etc |
| eval | False |  | use eval mode during test time. |
| num_test' | False | 50 | how many test images to run |
| batch_size | False | 1 | input batch size |
model='test'
load_size=parser.get_default('crop_size')
num_threads = 0
batch_size = 1
serial_batches = True
no_flip = True
display_id = -1
# Cyclegan Model
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| lambda_A | False | 10.0 | weight for cycle loss (A -> B -> A) |
| lambda_B | False | 10.0 | weight for cycle loss (B -> A -> B) |
| lambda_identity | False | 0.5 | use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1 |
# Test Model
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| model_suffix | False | '' | In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator. |
dataset_mode='single'
# Combine A and B Folder
| Argument | Required | Default | Help | 
| ------ | ------ | ------ | ------ |
| fold_A| False | ../dataset/50kshoes_edges | input directory for image A |
| fold_B | False | ../dataset/50kshoes_jpg | input directory for image B |
| fold_AB | False | ../dataset/test_AB | output directory |
| num_imgs | False | 1000000 | number of images |
| use_AB | False | | if true: (0001_A, 0001_B) to (0001_AB)|


