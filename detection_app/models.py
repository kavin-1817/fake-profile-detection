from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class Profile(models.Model):
    """Model to store user profile information"""
    bio = models.TextField(help_text="Profile bio text")
    friends_count = models.PositiveIntegerField(help_text="Number of friends/followers")
    total_posts = models.PositiveIntegerField(default=0, help_text="Total number of posts")
    account_age_days = models.PositiveIntegerField(default=0, help_text="Account age in days")
    profile_picture = models.BooleanField(default=True, help_text="Has profile picture")
    verified = models.BooleanField(default=False, help_text="Is verified account")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Profile {self.id} - {self.bio[:50]}..."

class DetectionResult(models.Model):
    """Model to store ML detection results"""
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='detections')
    is_fake = models.BooleanField(help_text="ML prediction: True if fake, False if genuine")
    confidence_score = models.FloatField(help_text="Confidence score (0-1)")
    model_version = models.CharField(max_length=50, default="v1.0")
    features_used = models.JSONField(default=dict, help_text="Features used for prediction")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        status = "FAKE" if self.is_fake else "GENUINE"
        return f"{status} - {self.profile} (Confidence: {self.confidence_score:.2f})"

class ModelMetrics(models.Model):
    """Model to store ML model performance metrics"""
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    confusion_matrix = models.JSONField(default=dict)
    training_date = models.DateTimeField(auto_now_add=True)
    dataset_size = models.PositiveIntegerField()
    
    class Meta:
        ordering = ['-training_date']
    
    def __str__(self):
        return f"{self.model_name} - Accuracy: {self.accuracy:.3f} ({self.training_date.strftime('%Y-%m-%d')})"

class FlaggedProfile(models.Model):
    """Model to track manually flagged profiles for review"""
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    flagged_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    reason = models.TextField(help_text="Reason for flagging")
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending Review'),
        ('confirmed', 'Confirmed Fake'),
        ('false_positive', 'False Positive'),
        ('resolved', 'Resolved')
    ], default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Flagged: {self.profile} - {self.status}"
