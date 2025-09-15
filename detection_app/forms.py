from django import forms
from .models import Profile, FlaggedProfile

class ProfileForm(forms.ModelForm):
    """Form for profile detection input"""
    
    class Meta:
        model = Profile
        fields = ['bio', 'friends_count', 'total_posts', 'account_age_days', 'verified', 'profile_picture']
        widgets = {
            'bio': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter profile bio text...',
                'required': True
            }),
            'friends_count': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Number of friends/followers',
                'min': 0,
                'required': True
            }),
            'total_posts': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Total number of posts',
                'min': 0,
                'required': True
            }),
            'account_age_days': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Account age in days (optional)',
                'min': 0,
                'value': 0
            }),
            'verified': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'profile_picture': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }
        labels = {
            'bio': 'Profile Bio',
            'friends_count': 'Friends/Followers Count',
            'total_posts': 'Total Number of Posts',
            'account_age_days': 'Account Age (Days)',
            'verified': 'Verified Account',
            'profile_picture': 'Has Profile Picture'
        }
        help_texts = {
            'bio': 'Enter the profile bio text to analyze',
            'friends_count': 'Number of friends or followers',
            'total_posts': 'Total number of posts made by the profile',
            'account_age_days': 'Age of the account in days (optional)',
            'verified': 'Check if this is a verified account',
            'profile_picture': 'Check if the profile has a picture'
        }

class FlagProfileForm(forms.ModelForm):
    """Form for flagging profiles"""
    
    class Meta:
        model = FlaggedProfile
        fields = ['reason']
        widgets = {
            'reason': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Explain why you think this profile is fake...',
                'required': True
            })
        }
        labels = {
            'reason': 'Reason for Flagging'
        }
        help_texts = {
            'reason': 'Please provide a detailed explanation for why you believe this profile is fake'
        }
