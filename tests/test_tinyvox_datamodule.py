import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.hparams import Parameters


def test_tinyvox():
    # Parse parameters with TinyVox settings
    import sys
    sys.argv = [
        'test',
        '--custom_dataset', 'True',
        '--use_vad', 'False',
        '--dataset_path', '/scratch2/mlavechin/tinyvox/TinyVox',
        '--language', 'en',
        '--batch_size', '64',
        '--num_proc', '8',
    ]

    parameters = Parameters.parse()

    # Test datamodule
    from utils.agent_utils import get_datamodule
    datamodule = get_datamodule(parameters.data_param)

    # Setup and test
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    first_batch = next(iter(train_loader))

    print("✅ TinyVox dataset works!")
    print(f"Batch keys: {first_batch.keys()}")
    print(f"Audio shape: {first_batch['array'].shape}")
    print(f"First phonemes: {first_batch['phonemes'][0]}")


if __name__ == "__main__":
    test_tinyvox()