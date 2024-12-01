# PyTorch MNIST CNN with CI/CD Pipeline

A PyTorch implementation of a Convolutional Neural Network for MNIST digit classification with automated testing, training, and deployment pipeline. The project includes data augmentation visualization and comprehensive testing.

## Project Structure
project/
├── .github/
│ └── workflows/
│ └── ci-cd.yml # GitHub Actions workflow
├── src/
│ ├── train.py # CNN model and training logic
│ ├── test.py # Model testing and validation
│ └── deploy.py # Deployment script
├── sample_images/ # Augmented MNIST samples
│ └── augmented_samples_.png
├── requirements.txt # Project dependencies
├── .gitignore # Git ignore rules
└── README.md # Project documentation

### Model Features:
- Input: 28x28 grayscale images
- Two convolutional layers with batch normalization
- Max pooling for dimension reduction
- Dropout for regularization
- Less than 25,000 parameters
- Achieves >95% accuracy in one epoch


### Visualization
- Sample images are saved in `sample_images/` directory
- Shows both original and augmented versions
- Helps visualize the effects of augmentation
- Automatically generated during training

## Testing Requirements

The model must pass these tests:
1. **Parameter Count**: < 25,000 parameters
2. **Input Compatibility**: Handles 28x28 grayscale images
3. **Output Shape**: 10 classes (digits 0-9)
4. **Model Accuracy**: > 95% in one epoch

## CI/CD Pipeline

The GitHub Actions workflow automates:
1. Environment setup (Python 3.9)
2. Dependency installation
3. Model training
4. Validation tests
5. Model deployment
6. Artifact storage
