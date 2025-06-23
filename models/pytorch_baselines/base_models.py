import torch
import torch.nn as nn
import timm

class TimmBaselineClassifier(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, in_chans: int = 3, dropout_rate: float = 0.5):
        super().__init__()
        # Load pretrained backbone, without its original classifier
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans, global_pool='avg')

        # Get the number of features from the backbone's global pool output
        num_features = self.backbone.num_features

        # Define the classification head
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape is expected to be (batch_size, channels, height, width)
        features = self.backbone(x)
        # features shape after global_pool='avg' is (batch_size, num_features)
        output = self.head(features)
        return output.squeeze(-1) # Remove last dim to match BCELoss target

# Specific model factory functions
def MobileNetV2_PyTorch(dropout_rate=0.5, **kwargs):
    return TimmBaselineClassifier(model_name='mobilenetv2_100', dropout_rate=dropout_rate, **kwargs)

def MobileNetV3Small_PyTorch(dropout_rate=0.5, **kwargs):
    # Timm has various MobileNetV3 small versions, e.g., mobilenetv3_small_050, mobilenetv3_small_075, mobilenetv3_small_100
    # Assuming 'mobilenetv3_small_100' is a reasonable equivalent
    return TimmBaselineClassifier(model_name='mobilenetv3_small_100', dropout_rate=dropout_rate, **kwargs)

def EfficientNetB0_PyTorch(dropout_rate=0.5, **kwargs):
    return TimmBaselineClassifier(model_name='efficientnet_b0', dropout_rate=dropout_rate, **kwargs)

def EfficientNetB3_PyTorch(dropout_rate=0.5, **kwargs):
    return TimmBaselineClassifier(model_name='efficientnet_b3', dropout_rate=dropout_rate, **kwargs)

def ResNet50_PyTorch(dropout_rate=0.5, **kwargs):
    return TimmBaselineClassifier(model_name='resnet50', dropout_rate=dropout_rate, **kwargs)

def InceptionV3_PyTorch(dropout_rate=0.5, **kwargs):
    # InceptionV3 typically requires 299x299 input, but timm models adapt.
    # The training script will need to handle image size appropriately if not 224x224.
    # For now, assuming the SpectrogramDataset will resize to what the train script passes.
    return TimmBaselineClassifier(model_name='inception_v3', dropout_rate=dropout_rate, **kwargs)

def DenseNet121_PyTorch(dropout_rate=0.5, **kwargs):
    return TimmBaselineClassifier(model_name='densenet121', dropout_rate=dropout_rate, **kwargs)

if __name__ == '__main__':
    # Example usage (requires timm and torch installed)
    # These would typically be run in the training script environment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test MobileNetV2
    mnetv2 = MobileNetV2_PyTorch(pretrained=True).to(device)
    print("MobileNetV2_PyTorch loaded successfully.")
    dummy_input_mnetv2 = torch.randn(2, 3, 224, 224).to(device)
    output_mnetv2 = mnetv2(dummy_input_mnetv2)
    print(f"MobileNetV2 output shape: {output_mnetv2.shape}, output: {output_mnetv2}")

    # Test EfficientNetB0
    effb0 = EfficientNetB0_PyTorch(pretrained=True).to(device)
    print("\nEfficientNetB0_PyTorch loaded successfully.")
    dummy_input_effb0 = torch.randn(2, 3, 224, 224).to(device) # EfficientNetB0 default is 224
    output_effb0 = effb0(dummy_input_effb0)
    print(f"EfficientNetB0 output shape: {output_effb0.shape}, output: {output_effb0}")

    # Test InceptionV3
    # Note: InceptionV3 default input size is 299x299.
    # The SpectrogramDataset in train_pytorch.py resizes to args.img_size, which defaults to (224,224)
    # This should be fine as timm models often adapt, but for optimal performance, matching pretrained size is best.
    incepv3 = InceptionV3_PyTorch(pretrained=True).to(device)
    print("\nInceptionV3_PyTorch loaded successfully.")
    dummy_input_incepv3 = torch.randn(2, 3, 224, 224).to(device) # Test with 224 for consistency with current scripts
    # dummy_input_incepv3 = torch.randn(2, 3, 299, 299).to(device) # For native InceptionV3 size
    output_incepv3 = incepv3(dummy_input_incepv3)
    print(f"InceptionV3 output shape: {output_incepv3.shape}, output: {output_incepv3}")

    print("\nAll tested PyTorch baseline models loaded and gave output.")
