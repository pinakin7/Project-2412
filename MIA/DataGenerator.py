from utils import utils
from MIA import mia
import torch
from AttackModel.Discriminator import Discriminator
from AttackModel.Generator import Generator as GANGenerator
from utils.CSVLogger import CSVLogger
from DatasetGenerator.main import generate_dataset
from tqdm import tqdm
from PrivateModels.DPGAN.Generator import Generator as DPGANGenerator

device = utils.device
logger = CSVLogger(f"{mia.SAVE_LOGS_PATH}\logs.csv")





def perform_attack_original_data(attack_model, start_idx=0):
    attack_model = attack_model.to(device)

    test_data = generate_dataset(batch_size=1,train=False)

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_data, 0), desc=f"Performing Attack Original Data", total=len(test_data)):
            data = data[0]
            is_in_data = attack_model(data.to(device))
            logger.log('Original Data', start_idx+i, is_in_data.item())

def perform_attack_gan_data(attack_model, gan, start_idx=0):
    gan = gan.to(device)
    attack_model = attack_model.to(device)

    with torch.no_grad():
        for i in tqdm(range(10000), desc=f"Performing Attack GAN Data", total=10000):
            noise = torch.rand(1,utils.dpgan_noise_dim,1,1, device=device)
            img = gan(noise)
            is_in_data = attack_model(img)
            logger.log('GAN Data', start_idx+i, is_in_data.item())


def perform_attack_dpgan_data(attack_model, dpgan, start_idx=0):
    dpgan = dpgan.to(device)
    attack_model = attack_model.to(device)

    with torch.no_grad():
        for i in tqdm(range(10000), desc=f"Performing Attack DPGAN Data", total=10000):
            noise = torch.rand(1,utils.dpgan_noise_dim,1,1, device=device)
            img = dpgan(noise)
            is_in_data = attack_model(img)
            logger.log('DPGAN Data', start_idx+i, is_in_data.item())



if __name__ == "__main__":
    # Perfoming Attack on the Original Dataset #
    attack_model = Discriminator(num_channels=utils.num_channels, num_gpus=utils.num_gpus,
                                 num_filters=utils.num_attack_discriminator_filter)
    attack_model.load_state_dict(torch.load(mia.ATTACK_MODEL_PATH))
    attack_model.eval()
    perform_attack_original_data(attack_model, 0)

    # Performing attack on the GAN generated Dataset #
    gan = GANGenerator(num_channels=utils.num_channels, num_gpus=utils.num_gpus,
                    num_filters=utils.num_dpgan_generator_filter, noise_dims=utils.dpgan_noise_dim)
    gan.load_state_dict(torch.load(mia.GAN_GENERATOR_PATH))
    gan.eval()
    perform_attack_gan_data(attack_model, gan, 10000)

    # Performing attack on the GAN generated Dataset #
    dpgan = DPGANGenerator(num_channels=utils.num_channels, num_gpus=utils.num_gpus, num_filters=utils.num_dpgan_generator_filter, noise_dims=utils.dpgan_noise_dim)
    dpgan.load_state_dict(torch.load(mia.DPGAN_GENERATOR_PATH))
    dpgan.eval()
    perform_attack_dpgan_data(attack_model, dpgan, 20000)