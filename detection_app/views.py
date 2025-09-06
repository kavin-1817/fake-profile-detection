from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Count, Q
from django.utils import timezone
from datetime import datetime, timedelta
import pickle
import os
import numpy as np
import pandas as pd
import re
from .models import Profile, DetectionResult, ModelMetrics, FlaggedProfile
from .forms import ProfileForm, FlagProfileForm
import json

# Global variables for model data
model_data = None
model_loaded = False

def load_model():
    """Load the ML model and related data"""
    global model_data, model_loaded
    
    if not model_loaded:
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                model_loaded = True
                print(f"Model loaded successfully: {model_data['model_name']}")
            else:
                print("Model file not found. Please run train_model.py first.")
                model_data = None
        except Exception as e:
            print(f"Error loading model: {e}")
            model_data = None
    
    return model_data

def preprocess_text(text):
    """Preprocess text for ML model"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_features(text):
    """Extract additional text-based features"""
    if not text:
        return {
            'text_length': 0,
            'word_count': 0,
            'exclamation_count': 0,
            'caps_ratio': 0,
            'spam_words': 0
        }
    
    text = str(text)
    
    # Spam indicators
    spam_words = ['free', 'win', 'click', 'money', 'earn', 'guaranteed', 'urgent', 'follow', 'like', 'dm', 'bitcoin', 'crypto', 'investment', 'profit', 'scheme', 'pyramid', 'mlm']
    spam_word_count = sum(1 for word in spam_words if word in text.lower())
    
    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'spam_words': spam_word_count
    }

def predict_profile(bio, friends_count, posts_per_week, account_age_days=0, verified=False, profile_pic=True):
    """Predict if a profile is fake using the ML model"""
    model_data = load_model()
    
    if not model_data:
        return None, 0.0, "Model not available"
    
    try:
        # Preprocess text
        bio_processed = preprocess_text(bio)
        
        # Extract text features
        text_features = extract_text_features(bio_processed)
        
        # Prepare numeric features
        numeric_features = np.array([[
            friends_count,
            posts_per_week,
            account_age_days,
            int(verified),
            int(profile_pic),
            text_features['text_length'],
            text_features['word_count'],
            text_features['exclamation_count'],
            text_features['caps_ratio'],
            text_features['spam_words']
        ]])
        
        # Vectorize text
        text_vectorized = model_data['vectorizer'].transform([bio_processed])
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([text_vectorized, numeric_features])
        
        # Make prediction
        model = model_data['model']
        scaler = model_data.get('scaler')
        
        if scaler and model_data['model_name'] == 'SVM':
            combined_features = scaler.transform(combined_features.toarray())
        
        prediction = model.predict(combined_features)[0]
        confidence = model.predict_proba(combined_features)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        return prediction, confidence, "Success"
        
    except Exception as e:
        return None, 0.0, str(e)

def index(request):
    """Main page with profile detection form"""
    if request.method == "POST":
        form = ProfileForm(request.POST)
        if form.is_valid():
            # Get form data
            bio = form.cleaned_data['bio']
            friends_count = form.cleaned_data['friends_count']
            posts_per_week = form.cleaned_data['posts_per_week']
            account_age_days = form.cleaned_data.get('account_age_days', 0)
            verified = form.cleaned_data.get('verified', False)
            profile_pic = form.cleaned_data.get('profile_picture', True)
            
            # Make prediction
            prediction, confidence, error = predict_profile(
                bio, friends_count, posts_per_week, account_age_days, verified, profile_pic
            )
            
            if prediction is not None:
                # Save profile to database
                profile = Profile.objects.create(
                    bio=bio,
                    friends_count=friends_count,
                    posts_per_week=posts_per_week,
                    account_age_days=account_age_days,
                    verified=verified,
                    profile_picture=profile_pic
                )
                
                # Save detection result
                DetectionResult.objects.create(
                    profile=profile,
                    is_fake=bool(prediction),
                    confidence_score=confidence,
                    model_version=model_data['model_name'] if model_data else 'unknown',
                    features_used={
                        'text_length': len(bio),
                        'friends_count': friends_count,
                        'posts_per_week': posts_per_week,
                        'account_age_days': account_age_days,
                        'verified': verified,
                        'profile_picture': profile_pic
                    }
                )
                
                result = "Fake Profile" if prediction else "Genuine Profile"
                confidence_percent = confidence * 100
                
                return render(request, "detection_app/result.html", {
                    "result": result,
                    "confidence": confidence_percent,
                    "profile": profile,
                    "is_fake": bool(prediction),
                    "error": None
                })
            else:
                messages.error(request, f"Prediction failed: {error}")
    else:
        form = ProfileForm()
    
    return render(request, "detection_app/index.html", {"form": form})

def result(request, profile_id):
    """Display detection result for a specific profile"""
    profile = get_object_or_404(Profile, id=profile_id)
    detection = profile.detections.first()
    
    if not detection:
        messages.error(request, "No detection result found for this profile.")
        return redirect('detection_app:index')
    
    return render(request, "detection_app/result.html", {
        "result": "Fake Profile" if detection.is_fake else "Genuine Profile",
        "confidence": detection.confidence_score * 100,
        "profile": profile,
        "is_fake": detection.is_fake,
        "error": None
    })

def admin_dashboard(request):
    """Admin dashboard with analytics and flagged profiles"""
    if not request.user.is_staff:
        messages.error(request, "Access denied. Admin privileges required.")
        return redirect('detection_app:index')
    
    # Get statistics
    total_profiles = Profile.objects.count()
    fake_profiles = DetectionResult.objects.filter(is_fake=True).count()
    genuine_profiles = DetectionResult.objects.filter(is_fake=False).count()
    
    # Recent detections
    recent_detections = DetectionResult.objects.select_related('profile').order_by('-created_at')[:10]
    
    # Flagged profiles
    flagged_profiles = FlaggedProfile.objects.select_related('profile').filter(status='pending').order_by('-created_at')
    
    # Model metrics
    latest_metrics = ModelMetrics.objects.order_by('-training_date').first()
    
    # Daily statistics for the last 30 days
    thirty_days_ago = timezone.now() - timedelta(days=30)
    daily_stats = []
    
    for i in range(30):
        date = timezone.now() - timedelta(days=i)
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_detections = DetectionResult.objects.filter(created_at__range=[day_start, day_end])
        daily_stats.append({
            'date': date.strftime('%Y-%m-%d'),
            'total': day_detections.count(),
            'fake': day_detections.filter(is_fake=True).count(),
            'genuine': day_detections.filter(is_fake=False).count()
        })
    
    context = {
        'total_profiles': total_profiles,
        'fake_profiles': fake_profiles,
        'genuine_profiles': genuine_profiles,
        'recent_detections': recent_detections,
        'flagged_profiles': flagged_profiles,
        'latest_metrics': latest_metrics,
        'daily_stats': daily_stats,
        'fake_percentage': (fake_profiles / total_profiles * 100) if total_profiles > 0 else 0
    }
    
    return render(request, "detection_app/admin_dashboard.html", context)

def flag_profile(request, profile_id):
    """Flag a profile for manual review"""
    profile = get_object_or_404(Profile, id=profile_id)
    
    if request.method == "POST":
        form = FlagProfileForm(request.POST)
        if form.is_valid():
            FlaggedProfile.objects.create(
                profile=profile,
                flagged_by=request.user if request.user.is_authenticated else None,
                reason=form.cleaned_data['reason']
            )
            messages.success(request, "Profile flagged for review.")
            return redirect('detection_app:admin_dashboard')
    else:
        form = FlagProfileForm()
    
    return render(request, "detection_app/flag_profile.html", {
        'form': form,
        'profile': profile
    })

def resolve_flag(request, flag_id, status):
    """Resolve a flagged profile"""
    if not request.user.is_staff:
        messages.error(request, "Access denied.")
        return redirect('detection_app:admin_dashboard')
    
    flag = get_object_or_404(FlaggedProfile, id=flag_id)
    flag.status = status
    flag.resolved_at = timezone.now()
    flag.save()
    
    messages.success(request, f"Flag resolved as {status}.")
    return redirect('admin_dashboard')

def analytics_api(request):
    """API endpoint for analytics data"""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Access denied'}, status=403)
    
    # Get detection statistics by day
    thirty_days_ago = timezone.now() - timedelta(days=30)
    daily_stats = []
    
    for i in range(30):
        date = timezone.now() - timedelta(days=i)
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_detections = DetectionResult.objects.filter(created_at__range=[day_start, day_end])
        daily_stats.append({
            'date': date.strftime('%Y-%m-%d'),
            'total': day_detections.count(),
            'fake': day_detections.filter(is_fake=True).count(),
            'genuine': day_detections.filter(is_fake=False).count()
        })
    
    return JsonResponse({
        'daily_stats': daily_stats,
        'total_profiles': Profile.objects.count(),
        'fake_profiles': DetectionResult.objects.filter(is_fake=True).count(),
        'genuine_profiles': DetectionResult.objects.filter(is_fake=False).count()
    })

def model_status(request):
    """Check model status and performance"""
    model_data = load_model()
    
    if not model_data:
        return JsonResponse({
            'status': 'error',
            'message': 'Model not loaded'
        })
    
    # Get latest model metrics
    latest_metrics = ModelMetrics.objects.order_by('-training_date').first()
    
    return JsonResponse({
        'status': 'loaded',
        'model_name': model_data['model_name'],
        'metrics': {
            'accuracy': latest_metrics.accuracy if latest_metrics else 0,
            'precision': latest_metrics.precision if latest_metrics else 0,
            'recall': latest_metrics.recall if latest_metrics else 0,
            'f1_score': latest_metrics.f1_score if latest_metrics else 0,
            'training_date': latest_metrics.training_date.isoformat() if latest_metrics else None
        } if latest_metrics else None
    })
