
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)



def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out
class ClientNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ClientNet, self).__init__()
        self.in_planes = 16
        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
       # out = self.layer2(out)
       # out = self.layer3(out)
        # out = F.avg_pool2d(out, 8)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out
    
class simpleClientNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(simpleClientNet, self).__init__()
        self.in_planes = 16
        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 1, stride=1)
   

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.input_features = 16 * 32 * 16  # Corresponds to 16 channels, each 32x32
        self.output_features = 16 * 32 * 16  # Same as input features

        # Define the MLP layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_features, self.input_features),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(self.input_features, self.input_features),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(self.input_features, self.input_features),  # Third hidden layer
            nn.ReLU()   # Activation function
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # Flatten the input
        x = x.view(-1, self.input_features)  # Flatten the input
        # Pass through the MLP layers
        x = self.layers(x)
        return x.reshape(-1, 16, 32, 16)
    
    
class complexClientNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(complexClientNet, self).__init__()
        self.in_planes = 16
        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 5, stride=1)
   

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out





class ServerNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ServerNet, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
class AGNDecoder(nn.Module):
    def __init__(self):
        super(AGNDecoder, self).__init__()
        
        # Deepening the embedding processor
        self.embedding_processor = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Matching the channel size with the half image
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        # Enhancing the half image processing
        self.half_image_processor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        # Further process the concatenated features to refine the full image
        self.combined_processor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, half_image, embedding1, embedding2):
        # Process the embedding and the half image separately
        embedding = torch.cat([embedding1, embedding2], -1)
        processed_embedding = self.embedding_processor(embedding)
        processed_half_image = self.half_image_processor(half_image)

        # Concatenate the processed parts
        combined = torch.cat([processed_half_image, processed_embedding[:, :, :, :16]], dim=3)  # Using the correct portion for concatenation

        # Refine the combined image
        full_image = self.combined_processor(combined)
        return full_image
class AGNDecoder2(nn.Module):
    def __init__(self):
        super(AGNDecoder2, self).__init__()
        
        # Process the embedding to adjust its features suitable for concatenation
        self.embedding_processor = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Ensure the channel size is 3 to match the half image
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        # Process the half image to enhance its features
        self.half_image_processor = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        # Further process the concatenated features to refine the full image
        self.combined_processor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # Final layer to ensure the output has 3 channels
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )



    def forward(self, half_image, embedding1, embedding2):
        # Process the embedding and the half image separately
        embedding = torch.cat([embedding1, embedding2], -1)
        processed_embedding = self.embedding_processor(embedding)  # Output size will match the half image
        processed_half_image = self.half_image_processor(half_image)

        # Concatenate along the width dimension to form a complete image
        # The processed_embedding can be directly used since its size is intended to match the output spatially
        combined = torch.cat([processed_half_image, processed_embedding[:, :, :, :16]], dim=3)  # Using the leftmost 16 columns of the embedding

        # Refine the combined image
        full_image = self.combined_processor(combined)
        return full_image


        
class ClientNet2(nn.Module):
    def __init__(self, depth, num_classes):
        super(ClientNet2, self).__init__()
        self.in_planes = 16
        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []   

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, 8)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)

        return out

class fullDecoder(nn.Module):
    def __init__(self):
        super(fullDecoder, self).__init__()

        # Maintain spatial dimension, reduce channels to 32
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Maintain spatial dimension, reduce channels to 16
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        # Final layer to output 3 channels for RGB image
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = torch.tanh(self.conv3(x))  # Use sigmoid to ensure output is in [0,1]
        return x
    
class HalfWidthGenerator(nn.Module):
    def __init__(self):
        super(HalfWidthGenerator, self).__init__()
        # Input size: (btz, 1, 28, 28)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=1),
            # After this layer, the size will be (btz, 64, 28, 14) due to stride=(1, 2)
            nn.ReLU()
        )
        self.to_original_depth = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3, 3), stride=1, padding=1)
        # Output size: (btz, 1, 28, 14)

    def forward(self, x):
        x = self.features(x)
        x = torch.tanh(self.to_original_depth(x)) # Using sigmoid to ensure the output is between 0 and 1
        return x
    
class EnhancedDecoder(nn.Module):
    def __init__(self):
        super(EnhancedDecoder, self).__init__()
        # Initial convolution layers
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Skip connection layer
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)  # To match dimensions
        
        # Final layers for reconstruction
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)  # Using concatenated channels
        self.bn4 = nn.BatchNorm2d(16)
        self.conv_final = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Initial convolutions
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        
        # Preparing for skip connection
        x_skip = self.conv3(x2)
        
        # Skip connection - concatenate along the channel dimension
        x_concat = torch.cat((x_skip, x1), dim=1)  # Assuming x1's dimensions can be matched to x_skip's
        
        # Final convolutions
        x3 = F.relu(self.bn4(self.conv4(x_concat)))
        x_final = torch.sigmoid(self.conv_final(x3))  # Ensuring output is in [0,1]
        return x_final

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Upsample to an intermediate size
        self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
        
        # Refine the features
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=2, 
            padding=1
        )

        # Adjust channels and size to (batch size, 3, 32, 16)
        self.conv3 = nn.Conv2d(
            in_channels=16, 
            out_channels=3, 
            kernel_size=(3, 3), 
            stride=1, 
            padding=1
        )

        # Tanh activation function
        self.tanh = nn.Tanh()

        # Tanh activation function
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x

class Condition_decoder(nn.Module):
    def __init__(self):
        super(Condition_decoder, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Maintain spatial dimension, reduce channels to 16
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        # Final layer to output 3 channels for RGB image
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        
        self.proj = nn.Linear(10, 16 * 32 * 32)

    def label_proj(self, labels):
        x = F.one_hot(labels, num_classes=10).float()
        x = self.proj(x).view(-1, 16, 32, 32)
        return x
    
    def forward(self, x, labels):
    
        x = x + self.label_proj(labels)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = torch.tanh(self.conv3(x))  # Use sigmoid to ensure output is in [0,1]
        return x




class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(in_features=32 * 4 * 8, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.sigmoid(self.fc(x))
        return x

class CLSDiscriminator(nn.Module):
    def __init__(self, num_class):
        super(CLSDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=32 * 4 * 8, out_features=num_class * 2)
        self.relu = nn. ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        gan_x = self.sigmoid(x)
        return gan_x, x
    
    
    
class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(in_features= 16 * 8 * 8, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.sigmoid(self.fc(x))
        return x

import torch.nn as nn
import torch.nn.functional as F

class CLSDiscriminator2(nn.Module):
    def __init__(self, num_class):
        super(CLSDiscriminator2, self).__init__()
        
        # Increased model complexity for better feature extraction
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization for stability
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128) # Batch normalization
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization
        
        # Adaptive Pooling to handle variable input sizes and improve generalization
        self.adap_pool = nn.AdaptiveAvgPool2d((4, 8))
        
        # Adjusted linear layer dimensions according to the new conv layers and adaptive pooling
        self.fc = nn.Linear(in_features=64 * 4 * 8, out_features=num_class * 2)
        
        # Retain ReLU and Sigmoid activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Apply adaptive pooling before flattening
        x = self.adap_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = self.fc(x)
        gan_x = self.sigmoid(x)  # Assuming the first half is for GAN discrimination
     # The second half for classification
        return gan_x, x
