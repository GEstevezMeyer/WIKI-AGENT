from datasets import load_dataset, IterableDataset

def divide_dataset_shards(huggingface_dataset: str,version_dataset: str,
                          split: str,number_shards: int = 1) -> list[IterableDataset]:
    
    if version_dataset:
        dataset:IterableDataset = load_dataset(huggingface_dataset, version_dataset, streaming=True)[split]
    else:
        dataset:IterableDataset = load_dataset(huggingface_dataset, streaming=True)[split]

    return [dataset.shard(num_shards=number_shards, index=i)for i in range(number_shards)]