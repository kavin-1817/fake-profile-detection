from django.core.management.base import BaseCommand
from django.utils import timezone
from detection_app.models import ModelMetrics
import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_profile_detection.settings')
django.setup()

# Import the training function
from detection_app.train_model import train_models

class Command(BaseCommand):
    help = 'Train the ML model and save metrics to database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force retraining even if model exists',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting ML model training...')
        )
        
        try:
            # Check if model already exists
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ml_model.pkl')
            if os.path.exists(model_path) and not options['force']:
                self.stdout.write(
                    self.style.WARNING('Model already exists. Use --force to retrain.')
                )
                return
            
            # Train the model
            model_data, results = train_models()
            
            if model_data and results:
                # Save metrics to database
                best_model_name = model_data['model_name']
                best_results = results[best_model_name]
                
                ModelMetrics.objects.create(
                    model_name=best_model_name,
                    accuracy=best_results['accuracy'],
                    precision=best_results['precision'],
                    recall=best_results['recall'],
                    f1_score=best_results['f1'],
                    confusion_matrix=best_results['confusion_matrix'],
                    dataset_size=len(model_data.get('dataset', []))
                )
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Model training completed successfully!\n'
                        f'Model: {best_model_name}\n'
                        f'Accuracy: {best_results["accuracy"]:.3f}\n'
                        f'Precision: {best_results["precision"]:.3f}\n'
                        f'Recall: {best_results["recall"]:.3f}\n'
                        f'F1 Score: {best_results["f1"]:.3f}'
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR('Model training failed!')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during training: {str(e)}')
            )
