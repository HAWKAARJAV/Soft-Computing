import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ==========================================
# 1. EEG Branch (Conv1D + LSTM + Attention)
# ==========================================

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (Batch, Time, Hidden)
        attn_weights = F.softmax(self.attention(lstm_output).squeeze(2), dim=1)  # (Batch, Time)
        # Weighted sum of lstm_output along the time dimension
        context_vector = torch.sum(lstm_output * attn_weights.unsqueeze(2), dim=1) # (Batch, Hidden)
        return context_vector, attn_weights

class EEGBranch(nn.Module):
    def __init__(self, in_channels=30, hidden_dim=64, out_dim=128):
        super(EEGBranch, self).__init__()
        # Conv layer to extract local temporal-spatial patterns
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # LSTM Layer for temporal sequence modelling
        # Input to LSTM should be (Batch, Time, Features)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        
        # Self Attention over LSTM sequence (Bidirectional means hidden_dim * 2)
        self.attention = SelfAttention(hidden_dim * 2)
        
        # Projection to standard out_dim
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        # x input shape: (Batch, Channels=30, Time=500)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Reshape for LSTM: (Batch, Time, Channels)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attended, _ = self.attention(lstm_out)
        
        out = self.fc(attended)
        return F.relu(out)

# ==========================================
# 2. Audio Branch (MelSpec + 2D CNN)
# ==========================================

class AudioBranch(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64, out_dim=128):
        super(AudioBranch, self).__init__()
        # On-the-fly MelSpectrogram transform
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        
        # A lightweight 2D CNN to process the (1, n_mels, Time) spectrogram
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Pool frequencies, maintain time dimension
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(64)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        # x input shape: (Batch, 80000)
        with torch.no_grad(): # Melspec doesn't need gradients usually
            # MPS backend on Apple Silicon does not support FFT operations natively.
            # We explicitly force just this single transformation to compute on the CPU
            original_device = x.device
            x_cpu = x.cpu()
            melspec_cpu = self.melspec.cpu()
            
            spec = melspec_cpu(x_cpu) # -> (Batch, n_mels, time_steps)
            spec = spec.to(original_device) # Move back to MPS/CUDA
            
            # Add channel dim: (Batch, 1, n_mels, time_steps)
            spec = spec.unsqueeze(1)
            # Log melspectrogram for better scaling (add small epsilon for stability)
            spec = torch.log(spec + 1e-9)
            
        x = self.pool1(F.relu(self.bn1(self.conv1(spec))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))) # (B, 64, 1, Time)
        
        # Reshape to sequence (B, Time, Features)
        x = x.squeeze(2).permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        attended, _ = self.attention(lstm_out)
        
        out = F.relu(self.fc(attended))
        return out

# ==========================================
# 3. Video Branch (TimeDistributed CNN + Pooling)
# ==========================================

class VideoBranch(nn.Module):
    def __init__(self, out_dim=128):
        super(VideoBranch, self).__init__()
        # We will use a simple custom ResNet-style block for speed/simplicity
        # Input to each frame: (Channels=3, H=56, W=56)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(128)
        self.fc = nn.Linear(128, out_dim)

    def forward_frame(self, x):
        # x: (Batch*Frames, Channels, H, W)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        # x input shape: (Batch, Frames=25, C=3, H=56, W=56)
        b, f, c, h, w = x.shape
        
        # Flatten batch and frames to process all frames at once through CNN
        x = x.view(b * f, c, h, w)
        frame_feats = self.forward_frame(x) # (B*F, 64)
        
        # Reshape back to sequence
        frame_feats = frame_feats.view(b, f, -1) # (B, F, 64)
        
        # Temporal analysis via LSTM and Attention
        lstm_out, _ = self.lstm(frame_feats)
        attended, _ = self.attention(lstm_out)
        
        out = F.relu(self.fc(attended))
        return out

# ==========================================
# 4. Fusion and Classification Model
# ==========================================

class EAVMultimodalFusionNetwork(nn.Module):
    def __init__(self, num_classes=5, embedding_dim=128):
        super(EAVMultimodalFusionNetwork, self).__init__()
        
        self.eeg_branch = EEGBranch(out_dim=embedding_dim)
        self.aud_branch = AudioBranch(out_dim=embedding_dim)
        self.vis_branch = VideoBranch(out_dim=embedding_dim)
        
        # Fusion dimensionality = 3 * embedding_dim
        fusion_dim = embedding_dim * 3
        
        # Top-level MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def encode_modalities(self, eeg_data, aud_data, vis_data):
        """Return the existing 128-D expert representations without fusing them."""
        eeg_feat = self.eeg_branch(eeg_data) # (B, 128)
        aud_feat = self.aud_branch(aud_data) # (B, 128)
        vis_feat = self.vis_branch(vis_data) # (B, 128)

        return eeg_feat, aud_feat, vis_feat

    def classify_features(self, eeg_feat, aud_feat, vis_feat):
        """Run the original late-concatenation classifier."""
        # Late Fusion via Concatenation
        fused = torch.cat((eeg_feat, aud_feat, vis_feat), dim=1) # (B, 384)

        return self.classifier(fused)

    def forward_with_features(self, eeg_data, aud_data, vis_data):
        """Return baseline logits and expert features for auxiliary heads."""
        features = self.encode_modalities(eeg_data, aud_data, vis_data)
        logits = self.classify_features(*features)
        return logits, features

    def forward(self, eeg_data, aud_data, vis_data):
        # Keep the historical logits-only API and checkpoint architecture intact.
        logits, _ = self.forward_with_features(eeg_data, aud_data, vis_data)
        return logits

# Quick shape test to make sure everything links up correctly
if __name__ == "__main__":
    b = 4
    eeg = torch.randn(b, 30, 500)
    aud = torch.randn(b, 80000)
    vis = torch.randn(b, 25, 3, 56, 56)
    
    model = EAVMultimodalFusionNetwork()
    out = model(eeg, aud, vis)
    print("Output shape:", out.shape)
    assert out.shape == (b, 5)
    print("Forward pass successful!")
