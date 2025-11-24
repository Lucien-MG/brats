from collections import defaultdict
import csv


# Training loop
num_epochs = 10
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

print("Starting training...")
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)
    
    # Train
    train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_metrics = validate_epoch(model, valid_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    for key in train_metrics:
        history[f'train_{key}'].append(train_metrics[key])
        history[f'val_{key}'].append(val_metrics[key])
    
    # Log to CSV
    log_to_csv(epoch + 1, train_metrics, val_metrics)
    
    # Print epoch results
    print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    print(f'Train Dice: {train_metrics["dice_coef"]:.4f}, Val Dice: {val_metrics["dice_coef"]:.4f}')
    print(f'Train Acc: {train_metrics["accuracy"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
    
    # Save best model (equivalent to ModelCheckpoint)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, best_model_path)
        print(f'New best model saved with val_loss: {val_loss:.6f}')
    
    # Save checkpoint every epoch (equivalent to Keras ModelCheckpoint)
    checkpoint_path = f'model_{epoch+1:02d}-{val_loss:.6f}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'history': history
    }, checkpoint_path)

print("Model Trained!")