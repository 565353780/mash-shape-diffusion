from mash_shape_diffusion.Module.trainer import Trainer

def demo():
    model_file_path = "./output/pretrain-S/model_last.pth"

    trainer = Trainer()
    #trainer.loadModel(model_file_path, True)
    trainer.train()
    return True
