import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
import model
import os

import numpy as np

def mixup_data(x, y, alpha=0.2, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if config.USE_MIXUP:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, config.MIXUP_ALPHA, device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Accuracy calculation during mixup is tricky, we'll just use the argmax
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if config.USE_MIXUP:
            correct += (lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def run_training(train_loader, val_loader, num_classes, class_weights=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = model.build_model(num_classes).to(device)
    
    # Weighted CrossEntropyLoss with Label Smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    # AdamW Optimizer
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.LR_FACTOR, patience=config.SCHEDULER_PATIENCE
    )
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        train_loss, train_acc = train_one_epoch(net, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(net, val_loader, criterion, device)
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), os.path.join(config.MODELS_DIR, "best_model.pth"))
            print(f"Saved Best Model! (Acc: {val_acc:.4f})")
            patience_counter = 0 # Reset patience
        else:
            patience_counter += 1
            
        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    return net, history
