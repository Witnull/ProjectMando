import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class NodeClassificationLogger:
    def __init__(self, base_log_dir='NodeClassify_Log'):
        # Create session-specific directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(base_log_dir, f'session_{timestamp}')
        self.create_directories()
        
        # Initialize log containers
        self.train_logs = {}
        self.validation_logs = {}
        self.test_logs = {}
        self.metadata = {}
        
    def create_directories(self):
        """Create necessary directories for logging"""
        paths = [
            self.session_dir,
            os.path.join(self.session_dir, 'metrics'),
            os.path.join(self.session_dir, 'plots'),
            os.path.join(self.session_dir, 'models')
        ]
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
            
    def log_metadata(self, args):
        """Log training configuration and metadata"""
        self.metadata.update({
            'timestamp': datetime.now().isoformat(),
            'configuration': args
        })
        self._save_json('metadata.json', self.metadata)
        
    def log_training_stats(self, fold, epoch, stats):
        """Log training statistics for each epoch"""
        if fold not in self.train_logs:
            self.train_logs[fold] = []
        
        epoch_stats = {
            'epoch': epoch,
            **stats
        }
        self.train_logs[fold].append(epoch_stats)
        self._save_json(f'metrics/train_fold_{fold}.json', self.train_logs[fold])
        
    def log_validation_stats(self, fold, epoch, stats):
        """Log validation statistics for each epoch"""
        if fold not in self.validation_logs:
            self.validation_logs[fold] = []
            
        epoch_stats = {
            'epoch': epoch,
            **stats
        }
        self.validation_logs[fold].append(epoch_stats)
        self._save_json(f'metrics/val_fold_{fold}.json', self.validation_logs[fold])
        
    def log_test_results(self, fold, results):
        """Log test results"""
        self.test_logs[fold] = results
        self._save_json(f'metrics/test_fold_{fold}.json', results)
        
    def plot_training_curves(self, fold):
        """Generate and save training curves"""
        metrics = ['loss', 'acc', 'micro_f1', 'macro_f1', 'buggy_f1']
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Plot training data
            train_data = [log[metric] for log in self.train_logs[fold]]
            plt.plot(train_data, label=f'Train {metric}')
            
            # Plot validation data
            val_data = [log[metric] for log in self.validation_logs[fold]]
            plt.plot(val_data, label=f'Validation {metric}')
            
            plt.title(f'Fold {fold} - {metric} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            
            save_path = os.path.join(self.session_dir, 'plots', f'fold_{fold}_{metric}.png')
            plt.savefig(save_path)
            plt.close()
            
    def plot_confusion_matrix(self, fold, confusion_matrix, labels=None):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels if labels else 'auto',
                   yticklabels=labels if labels else 'auto')
        plt.title(f'Fold {fold} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = os.path.join(self.session_dir, 'plots', f'fold_{fold}_confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        
    def _save_json(self, filename, data):
        """Helper method to save JSON data"""
        file_path = os.path.join(self.session_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
