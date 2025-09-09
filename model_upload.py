import kagglehub
from kagglehub.config import get_kaggle_credentials

# Other ways to authenticate also available: https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate
kagglehub.login() 

username = "sairamnarendrababu"
kagglehub.model_upload(f'{username}/my_model/pyTorch/2b', 'best-model-epoch=16-val_loss=5.45.ckpt', 'Apache 2.0')

# Run the same command again to upload a new version for an existing variation.