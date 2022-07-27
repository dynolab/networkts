import os
import sys
import logging

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, call


LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../config', config_name='config')
def main(cfg: DictConfig) -> None:
    LOGGER.info(f'{os.getcwd()}')
    # Append the rep root just in case we are running directly from the rep
    sys.path.append(hydra.utils.get_original_cwd())

    # Main code
    forecaster = instantiate(cfg.forecaster)
    decomp = instantiate(cfg.decomposition)
    dataset = call(cfg.dataset)
    print(OmegaConf.to_yaml(cfg))
    LOGGER.info(f'{os.getcwd()}')


if __name__ == '__main__':
    main()
