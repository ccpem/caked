from .base import AbstractDataLoader

class DiskDataLoader(AbstractDataLoader):
    def __init__(self, pipeline: str, classes: list[str], dataset_size: int, save_to_disk: bool, training: bool) -> None:
        super().__init__(pipeline, classes, dataset_size, save_to_disk, training)
    
    def load(self):
        return super().load()
    
    def process(self):
        return super().process()
    
    def get_loader(self, split_size: float, batch_size: int):
        return super().get_loader(split_size, batch_size)