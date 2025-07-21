# Installation

```shell
conda env create -f conda_environment.yml
conda activate phorec
pip install -r requirements.txt
pip install pytorch_lightning==1.5.10 lightning-bolts==0.5.0
pip install transformers --upgrade
```

Train:

```shell
WANDB_MODE=disabled python main.py --gpu 1 --num_workers 2 --language ru --subset ru --network_name WavLM --train True --num_proc 1 --enable_progress_bar False --lr 2e-2
```

Test: