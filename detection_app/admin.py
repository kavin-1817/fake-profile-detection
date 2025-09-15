from django.contrib import admin
from .models import Profile, DetectionResult, ModelMetrics, FlaggedProfile

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ['id', 'bio_preview', 'friends_count', 'total_posts', 'account_age_days', 'verified', 'created_at']
    list_filter = ['verified', 'profile_picture', 'created_at']
    search_fields = ['bio']
    readonly_fields = ['created_at']
    
    def bio_preview(self, obj):
        return obj.bio[:50] + "..." if len(obj.bio) > 50 else obj.bio
    bio_preview.short_description = 'Bio Preview'

@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ['profile', 'is_fake', 'confidence_score', 'model_version', 'created_at']
    list_filter = ['is_fake', 'model_version', 'created_at']
    search_fields = ['profile__bio']
    readonly_fields = ['created_at']

@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'dataset_size', 'training_date']
    list_filter = ['model_name', 'training_date']
    readonly_fields = ['training_date']

@admin.register(FlaggedProfile)
class FlaggedProfileAdmin(admin.ModelAdmin):
    list_display = ['profile', 'status', 'flagged_by', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['profile__bio', 'reason']
    readonly_fields = ['created_at']
