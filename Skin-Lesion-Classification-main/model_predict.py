#!/usr/bin/env python3
"""
HAM10000 Skin Lesion Classification Model - DEBUG VERSION
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HAM10000 Classes (exactly from your project)
CLASSES = [
    'actinic keratosis',
    'basal cell carcinoma', 
    'benign keratosis',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'vascular lesion'
]

# Severity mapping (exactly from your project)
SEVERITY_MAPPING = {
    'melanoma': 5,
    'basal cell carcinoma': 4,
    'actinic keratosis': 3,
    'vascular lesion': 2,
    'dermatofibroma': 2,
    'benign keratosis': 2,
    'nevus': 1
}

# Your exact model class from Colab
class MultiTaskSkinLesionModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='resnet50'):
        super(MultiTaskSkinLesionModel, self).__init__()

        # Load the base model
        if model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.features = base_model.features
            self.feature_dim = 1280
        elif model_name == 'densenet121':
            base_model = models.densenet121(pretrained=pretrained)
            self.features = base_model.features
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Common layers
        self.dropout = nn.Dropout(0.5)

        # Classification branch
        self.classification = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Severity estimation branch (regression)
        self.severity = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        # Classification output
        class_output = self.classification(x)

        # Severity estimation output
        severity_output = self.severity(x)

        return class_output, severity_output

def load_model():
    """Load the trained model with extensive debugging"""
    try:
        print("🔧 DEBUG: Loading HAM10000 model...", file=sys.stderr)
        
        # Initialize model architecture
        model = MultiTaskSkinLesionModel(
            num_classes=len(CLASSES), 
            pretrained=False,  # Important: False when loading trained weights
            model_name='resnet50'
        )
        
        print(f"🔧 DEBUG: Model architecture created with {len(CLASSES)} classes", file=sys.stderr)
        
        # Check model file
        model_path = 'best_skin_lesion_model_ham10000.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"🔧 DEBUG: Model file size: {file_size:.1f} MB", file=sys.stderr)
        
        # Load state dict
        print("🔧 DEBUG: Loading state dictionary...", file=sys.stderr)
        state_dict = torch.load(model_path, map_location=device)
        
        # Debug: Check state dict keys
        print(f"🔧 DEBUG: State dict has {len(state_dict)} parameters", file=sys.stderr)
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"🔧 DEBUG: Model loaded successfully on {device}", file=sys.stderr)
        
        # TEST THE MODEL with random input
        print("🔧 DEBUG: Testing model with random input...", file=sys.stderr)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(device)
            class_output, severity_output = model(test_input)
            test_probs = torch.nn.functional.softmax(class_output, dim=1)[0]
            
            print("🔧 DEBUG: Random input test results:", file=sys.stderr)
            for i, class_name in enumerate(CLASSES):
                print(f"   {class_name}: {test_probs[i]:.4f} ({test_probs[i]*100:.1f}%)", file=sys.stderr)
            
            top_class_idx = torch.argmax(test_probs).item()
            print(f"🔧 DEBUG: Top random prediction: {CLASSES[top_class_idx]} ({test_probs[top_class_idx]:.2%})", file=sys.stderr)
        
        return model
        
    except Exception as e:
        print(f"❌ DEBUG: Error loading model: {e}", file=sys.stderr)
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr)
        return None

def preprocess_image(image_path):
    """Preprocess image with debugging"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        print(f"🔧 DEBUG: Original image size: {image.size}", file=sys.stderr)
        print(f"🔧 DEBUG: Image mode: {image.mode}", file=sys.stderr)
        
        # Apply exact same transforms as validation dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        print(f"🔧 DEBUG: Processed tensor shape: {image_tensor.shape}", file=sys.stderr)
        print(f"🔧 DEBUG: Tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]", file=sys.stderr)
        print(f"🔧 DEBUG: Tensor mean: {image_tensor.mean():.3f}, std: {image_tensor.std():.3f}", file=sys.stderr)
        
        return image_tensor.to(device)
        
    except Exception as e:
        print(f"❌ DEBUG: Error preprocessing image: {e}", file=sys.stderr)
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr)
        return None

def predict_image(image_path):
    """Run prediction with extensive debugging"""
    try:
        # Load model
        model = load_model()
        if model is None:
            return {
                'success': False,
                'error': 'Failed to load model'
            }
        
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            return {
                'success': False,
                'error': 'Failed to preprocess image'
            }
        
        print("🔧 DEBUG: Running prediction...", file=sys.stderr)
        
        # Run inference
        with torch.no_grad():
            class_output, severity_output = model(image_tensor)
            
            # DEBUG: Print raw outputs
            raw_outputs = class_output[0].detach().cpu().numpy()
            print(f"🔧 DEBUG: Raw class outputs: {raw_outputs}", file=sys.stderr)
            print(f"🔧 DEBUG: Raw output range: [{raw_outputs.min():.3f}, {raw_outputs.max():.3f}]", file=sys.stderr)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(class_output, dim=1)[0]
            
            # DEBUG: Print all probabilities
            print("🔧 DEBUG: All class probabilities:", file=sys.stderr)
            for i, class_name in enumerate(CLASSES):
                print(f"   {class_name}: {probabilities[i]:.4f} ({probabilities[i]*100:.1f}%)", file=sys.stderr)
            
            # Get predicted class
            predicted_class_idx = torch.argmax(probabilities).item()
            predicted_class = CLASSES[predicted_class_idx]
            
            print(f"🔧 DEBUG: Predicted class index: {predicted_class_idx}", file=sys.stderr)
            print(f"🔧 DEBUG: Predicted class name: {predicted_class}", file=sys.stderr)
            
            # Get severity prediction
            severity_normalized = severity_output.item()
            severity_score = severity_normalized * 4 + 1
            
            print(f"🔧 DEBUG: Raw severity output: {severity_normalized:.4f}", file=sys.stderr)
            print(f"🔧 DEBUG: Converted severity score: {severity_score:.2f}", file=sys.stderr)
            
            # Create predictions list
            predictions = []
            for i, class_name in enumerate(CLASSES):
                predictions.append({
                    'class': class_name,
                    'confidence': float(probabilities[i])
                })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Get medical severity level
            medical_severity = SEVERITY_MAPPING.get(predicted_class, 2)
            
            # Check if this looks like a reasonable prediction
            max_confidence = float(torch.max(probabilities))
            second_max = float(torch.topk(probabilities, 2)[0][1])
            confidence_gap = max_confidence - second_max
            
            print(f"🔧 DEBUG: Max confidence: {max_confidence:.4f}", file=sys.stderr)
            print(f"🔧 DEBUG: Second max: {second_max:.4f}", file=sys.stderr)
            print(f"🔧 DEBUG: Confidence gap: {confidence_gap:.4f}", file=sys.stderr)
            
            if max_confidence > 0.8 and predicted_class == 'nevus':
                print("⚠️ DEBUG: High confidence nevus prediction - might indicate model bias!", file=sys.stderr)
            
            # Prepare response
            result = {
                'success': True,
                'predictions': predictions,
                'primary_prediction': {
                    'class': predicted_class,
                    'confidence': float(probabilities[predicted_class_idx])
                },
                'severity': {
                    'score': float(severity_score),
                    'medical_level': medical_severity,
                    'predicted_class': predicted_class
                },
                'debug_info': {
                    'raw_outputs': raw_outputs.tolist(),
                    'all_probabilities': [float(p) for p in probabilities],
                    'confidence_gap': float(confidence_gap),
                    'max_confidence': float(max_confidence)
                }
            }
            
            print(f"🔧 DEBUG: Final prediction: {predicted_class} ({probabilities[predicted_class_idx]:.1%})", file=sys.stderr)
            
            return result
            
    except Exception as e:
        print(f"❌ DEBUG: Prediction error: {e}", file=sys.stderr)
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr)
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 2:
        result = {
            'success': False,
            'error': 'Usage: python model_predict.py <image_path>'
        }
        print(json.dumps(result))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"🔧 DEBUG: Processing image: {image_path}", file=sys.stderr)
    print(f"🔧 DEBUG: Python version: {sys.version}", file=sys.stderr)
    print(f"🔧 DEBUG: PyTorch version: {torch.__version__}", file=sys.stderr)
    print(f"🔧 DEBUG: Device: {device}", file=sys.stderr)
    
    # Run prediction
    result = predict_image(image_path)
    
    # Output JSON result to stdout (Node.js will read this)
    print(json.dumps(result))

if __name__ == '__main__':
    main()