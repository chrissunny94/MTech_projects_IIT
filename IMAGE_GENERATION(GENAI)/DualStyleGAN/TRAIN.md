## (3) Training DualStyleGAN

Download the supporting models to the `./checkpoint/` folder:

| Model | Description |
| :--- | :--- |
| [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) | StyleGAN model trained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch). |
| [model_ir_se50.pth](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for ID loss. |

### Facial Destylization

**Step 1: Prepare data.** 

Prepare the dataset in `./data/DATASET_NAME/images/train/`. First create lmdb datasets:

``` 
python ./model/stylegan/prepare_data.py --out ./data/simpsons/lmdb/ --n_worker 4 --size 1024 ./data/simpsons/images/

```


**Step 2: Fine-tune StyleGAN.** Fine-tune StyleGAN in distributed settings:

```
export TORCH_CUDA_ARCH_LIST="8.6"

```

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 finetune_stylegan.py --iter 600
                          --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon
                          --augment ./data/cartoon/lmdb/
```


The fine-tuned model can be found in `./checkpoint/cartoon/finetune-000600.pt`. Intermediate results are saved in `./log/cartoon/`.


```
WORKSPACE/MTech_projects_IIT/IMAGE_GENERATION(GENAI)/DualStyleGAN$ python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 finetune_stylegan.py --iter 600
                          --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon
                          --augment ./data/cartoon/lmdb/
/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/torch/distributed/launch.py:208: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use-env is set by default in torchrun.
If your script expects `--local-rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  main()
W0118 22:00:09.772063 138404564240192 torch/distributed/run.py:779] 
W0118 22:00:09.772063 138404564240192 torch/distributed/run.py:779] *****************************************
W0118 22:00:09.772063 138404564240192 torch/distributed/run.py:779] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0118 22:00:09.772063 138404564240192 torch/distributed/run.py:779] *****************************************
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
usage: finetune_stylegan.py [-h] [--iter ITER] [--batch BATCH] [--n_sample N_SAMPLE] [--size SIZE] [--r1 R1] [--path_regularize PATH_REGULARIZE] [--path_batch_shrink PATH_BATCH_SHRINK] [--d_reg_every D_REG_EVERY]
                            [--g_reg_every G_REG_EVERY] [--mixing MIXING] [--ckpt CKPT] [--lr LR] [--channel_multiplier CHANNEL_MULTIPLIER] [--wandb] [--local_rank LOCAL_RANK] [--augment] [--augment_p AUGMENT_P]
                            [--ada_target ADA_TARGET] [--ada_length ADA_LENGTH] [--ada_every ADA_EVERY] [--save_every SAVE_EVERY] [--style STYLE] [--model_path MODEL_PATH]
                            path
finetune_stylegan.py: error: the following arguments are required: path
W0118 22:00:34.169929 138404564240192 torch/distributed/elastic/multiprocessing/api.py:858] Sending process 3293502 closing signal SIGTERM
W0118 22:00:34.170196 138404564240192 torch/distributed/elastic/multiprocessing/api.py:858] Sending process 3293505 closing signal SIGTERM
W0118 22:00:34.170255 138404564240192 torch/distributed/elastic/multiprocessing/api.py:858] Sending process 3293507 closing signal SIGTERM
E0118 22:00:34.185714 138404564240192 torch/distributed/elastic/multiprocessing/api.py:833] failed (exitcode: 2) local_rank: 0 (pid: 3293501) of binary: /media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/bin/python
Traceback (most recent call last):
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/torch/distributed/launch.py", line 208, in <module>
    main()
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/typing_extensions.py", line 2853, in wrapper
    return arg(*args, **kwargs)
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/torch/distributed/launch.py", line 204, in main
    launch(args)
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/torch/distributed/launch.py", line 189, in launch
    run(args)
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/torch/distributed/run.py", line 892, in run
    elastic_launch(
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 133, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/media/chris/UBUNTU_PARTITION/anaconda3/envs/dualstylegan_env/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 264, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
finetune_stylegan.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2025-01-18_22:00:34
  host      : chris-B760M-DS3H
  rank      : 2 (local_rank: 2)
  exitcode  : 2 (pid: 3293503)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2025-01-18_22:00:34
  host      : chris-B760M-DS3H
  rank      : 3 (local_rank: 3)
  exitcode  : 2 (pid: 3293504)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2025-01-18_22:00:34
  host      : chris-B760M-DS3H
  rank      : 5 (local_rank: 5)
  exitcode  : 2 (pid: 3293506)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[4]:
  time      : 2025-01-18_22:00:34
  host      : chris-B760M-DS3H
  rank      : 7 (local_rank: 7)
  exitcode  : 2 (pid: 3293509)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-01-18_22:00:34
  host      : chris-B760M-DS3H
  rank      : 0 (local_rank: 0)
  exitcode  : 2 (pid: 3293501)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
--batch: command not found
--augment: command not found
```


```
torchrun --nproc_per_node=1 --master_port=8765 finetune_stylegan.py --iter 5 --batch 4 --ckpt ./checkpoint/stylegan2-ffhq-config-f.pt --style cartoon --augment ./data/simpsons/lmdb/

```


**Step 3: Destylize artistic portraits.** 
```python
python destylize.py --model_name FINETUNED_MODEL_NAME --batch BATCH_SIZE --iter ITERATIONS DATASET_NAME
```
Take the cartoon dataset for example, run:
> python destylize.py --model_name finetune-000600.pt --batch 1 --iter 300 cartoon

The intrinsic and extrinsic style codes are saved in `./checkpoint/cartoon/instyle_code.npy` and `./checkpoint/cartoon/exstyle_code.npy`, respectively. Intermediate results are saved in `./log/cartoon/destylization/`.
To speed up destylization, set `--batch` to large value like 16. 
For styles severely different from real faces, set `--truncation` to small value like 0.5 to make the results more photo-realistic (it enables DualStyleGAN to learn larger structrue deformations).


### Progressive Fine-Tuning 

**Stage 1 & 2: Pretrain DualStyleGAN on FFHQ.** 
We provide our pretrained model [generator-pretrain.pt](https://drive.google.com/file/d/1j8sIvQZYW5rZ0v1SDMn2VEJFqfRjMW3f/view?usp=sharing) at [Google Drive](https://drive.google.com/drive/folders/1GZQ6Gs5AzJq9lUL-ldIQexi0JYPKNy8b?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1sOpPszHfHSgFsgw47S6aAA ) (access code: cvpr). This model is obtained by:
> python -m torch.distributed.launch --nproc_per_node=1 --master_port=8765 pretrain_dualstylegan.py --iter 3000
                          --batch 4 ./data/ffhq/lmdb/

where `./data/ffhq/lmdb/` contains the lmdb data created from the FFHQ dataset via `./model/stylegan/prepare_data.py`.

**Stage 3: Fine-Tune DualStyleGAN on Target Domain.** Fine-tune DualStyleGAN in distributed settings:
```python
python -m torch.distributed.launch --nproc_per_node=N_GPU --master_port=PORT finetune_dualstylegan.py --iter ITERATIONS \ 
                          --batch BATCH_SIZE --ckpt PRETRAINED_MODEL_PATH --augment DATASET_NAME
```
The loss term weights can be specified by `--style_loss` (λ<sub>FM</sub>), `--CX_loss` (λ<sub>CX</sub>), `--perc_loss` (λ<sub>perc</sub>), `--id_loss` (λ<sub>ID</sub>) and `--L2_reg_loss` (λ<sub>reg</sub>). λ<sub>ID</sub> and λ<sub>reg</sub> are suggested to be tuned for each style dataset to achieve ideal performance. More options can be found via `python finetune_dualstylegan.py -h`.

Take the Cartoon dataset as an example, run (multi-GPU enables a large batch size of 8\*4=32 for better performance):
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=8765 finetune_dualstylegan.py --iter 1500 --batch 4 --ckpt ./checkpoint/generator-pretrain.pt 
--style_loss 0.25 --CX_loss 0.25 --perc_loss 1 --id_loss 1 --L2_reg_loss 0.015 --augment cartoon

The fine-tuned models can be found in `./checkpoint/cartoon/generator-ITER.pt` where ITER = 001000, 001100, ..., 001500. Intermediate results are saved in `./log/cartoon/`. Large ITER has strong cartoon styles but at the cost of artifacts, and users may select the most balanced one from 1000-1500. We use 1400 for our paper experiments.
