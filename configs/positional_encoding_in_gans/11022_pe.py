_base_ = ['../singan/11022.py']

num_scales = 11  # start from zero
model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE',
        num_scales=num_scales,
        padding=1,
        pad_at_head=False,
        first_stage_in_channels=2,
        positional_encoding=dict(type='CSG')),
    discriminator=dict(num_scales=num_scales))

train_cfg = dict(first_fixed_noises_ch=2)

data = dict(
    train=dict(
        img_path='/home/uhrgan/pe/mmgeneration/SCENERY_432/11022_phase_3_w_432.jpg',
        min_size=25,
        max_size=432,
    ))

dist_params = dict(backend='nccl')
total_iters = 24000
