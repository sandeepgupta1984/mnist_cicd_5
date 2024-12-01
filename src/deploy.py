import os
import glob
import shutil
from datetime import datetime

def deploy_model():
    # Create deployment directory
    deploy_dir = 'deployed_models'
    os.makedirs(deploy_dir, exist_ok=True)

    # Find the latest model file
    model_files = glob.glob('model_*.pt')
    if not model_files:
        raise Exception("No model files found for deployment")
    
    latest_model = max(model_files)
    
    # Copy to deployment directory
    deploy_path = os.path.join(deploy_dir, latest_model)
    shutil.copy2(latest_model, deploy_path)
    print(f"Model deployed to {deploy_path}")

if __name__ == "__main__":
    deploy_model() 