import pandas as pd
import numpy as np
import nltk
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

def create_synthetic_dataset():
    """Create a comprehensive synthetic dataset for fake profile detection"""
    
    # Genuine profiles
    genuine_profiles = [
        # Students and professionals
        {"bio": "Computer Science student at MIT. Love coding and machine learning. Coffee enthusiast.", "friends": 450, "posts": 3, "age_days": 1200, "verified": False, "profile_pic": True},
        {"bio": "Software Engineer at Google. Passionate about AI and open source projects.", "friends": 1200, "posts": 2, "age_days": 2000, "verified": True, "profile_pic": True},
        {"bio": "Marketing professional. Love traveling and photography. Based in NYC.", "friends": 800, "posts": 5, "age_days": 1500, "verified": False, "profile_pic": True},
        {"bio": "Medical student. Future doctor. Love helping people and learning new things.", "friends": 300, "posts": 4, "age_days": 800, "verified": False, "profile_pic": True},
        {"bio": "Artist and designer. Creating beautiful things every day. Follow for art updates!", "friends": 2500, "posts": 8, "age_days": 3000, "verified": True, "profile_pic": True},
        {"bio": "Teacher and mom of two. Sharing educational content and parenting tips.", "friends": 600, "posts": 6, "age_days": 1800, "verified": False, "profile_pic": True},
        {"bio": "Fitness trainer and nutritionist. Helping people achieve their health goals.", "friends": 1500, "posts": 7, "age_days": 2200, "verified": False, "profile_pic": True},
        {"bio": "PhD in Physics. Research scientist. Love quantum mechanics and space exploration.", "friends": 400, "posts": 2, "age_days": 2500, "verified": False, "profile_pic": True},
        {"bio": "Chef and food blogger. Creating delicious recipes and sharing culinary adventures.", "friends": 3000, "posts": 10, "age_days": 4000, "verified": True, "profile_pic": True},
        {"bio": "Environmental activist. Fighting climate change one post at a time.", "friends": 2000, "posts": 12, "age_days": 1600, "verified": False, "profile_pic": True},
    ]
    
    # Fake profiles
    fake_profiles = [
        # Spam and promotional accounts
        {"bio": "CLICK HERE TO WIN $1000!!! FREE MONEY!!! NO SCAM!!!", "friends": 5, "posts": 50, "age_days": 10, "verified": False, "profile_pic": False},
        {"bio": "Make money fast! Work from home! Earn $5000 per week! DM me now!", "friends": 2, "posts": 30, "age_days": 5, "verified": False, "profile_pic": False},
        {"bio": "FREE BITCOIN!!! INVEST NOW!!! GUARANTEED PROFITS!!!", "friends": 1, "posts": 25, "age_days": 3, "verified": False, "profile_pic": False},
        {"bio": "Hot singles in your area! Click now! Adult content!", "friends": 0, "posts": 40, "age_days": 7, "verified": False, "profile_pic": False},
        {"bio": "FOLLOW FOR FOLLOW!!! LIKE FOR LIKE!!! INSTANT FOLLOWERS!!!", "friends": 50, "posts": 100, "age_days": 15, "verified": False, "profile_pic": True},
        {"bio": "Buy followers! Buy likes! Social media marketing services!", "friends": 20, "posts": 60, "age_days": 20, "verified": False, "profile_pic": False},
        {"bio": "WIN IPHONE 15!!! CLICK LINK!!! VERIFY ACCOUNT!!!", "friends": 3, "posts": 35, "age_days": 8, "verified": False, "profile_pic": False},
        {"bio": "Crypto investment opportunity! 1000% returns! Join our group!", "friends": 8, "posts": 45, "age_days": 12, "verified": False, "profile_pic": False},
        {"bio": "URGENT!!! HELP NEEDED!!! SEND MONEY!!! FAMILY EMERGENCY!!!", "friends": 1, "posts": 20, "age_days": 4, "verified": False, "profile_pic": False},
        {"bio": "Get rich quick scheme! Pyramid marketing! Join now!", "friends": 15, "posts": 55, "age_days": 25, "verified": False, "profile_pic": True},
    ]
    
    # Create more variations
    all_profiles = []
    
    # Add genuine profiles
    for profile in genuine_profiles:
        all_profiles.append({**profile, "label": 0})
    
    # Add fake profiles
    for profile in fake_profiles:
        all_profiles.append({**profile, "label": 1})
    
    # Generate more synthetic data
    for i in range(20):
        # More genuine profiles
        genuine_variations = [
            f"University student studying {['Computer Science', 'Medicine', 'Engineering', 'Business'][i%4]}. Love learning!",
            f"Professional {['developer', 'designer', 'marketer', 'consultant'][i%4]} with {5+i} years experience.",
            f"Passionate about {['technology', 'art', 'music', 'sports'][i%4]}. Always exploring new things!",
            f"Working in {['tech', 'finance', 'healthcare', 'education'][i%4]} industry. Love my job!",
        ]
        
        all_profiles.append({
            "bio": genuine_variations[i % len(genuine_variations)],
            "friends": np.random.randint(200, 2000),
            "posts": np.random.randint(1, 10),
            "age_days": np.random.randint(500, 3000),
            "verified": np.random.choice([True, False], p=[0.1, 0.9]),
            "profile_pic": True,
            "label": 0
        })
        
        # More fake profiles
        fake_variations = [
            f"EARN MONEY FAST!!! {['Bitcoin', 'Stocks', 'Forex', 'Crypto'][i%4]} INVESTMENT!!!",
            f"FREE {['iPhone', 'MacBook', 'Cash', 'Gift Cards'][i%4]}!!! CLICK NOW!!!",
            f"WORK FROM HOME!!! EARN ${np.random.randint(1000, 10000)} PER WEEK!!!",
            f"FOLLOW FOR FOLLOW!!! {['LIKES', 'SHARES', 'COMMENTS', 'VIEWS'][i%4]}!!!",
        ]
        
        all_profiles.append({
            "bio": fake_variations[i % len(fake_variations)],
            "friends": np.random.randint(0, 50),
            "posts": np.random.randint(20, 100),
            "age_days": np.random.randint(1, 30),
            "verified": False,
            "profile_pic": np.random.choice([True, False], p=[0.3, 0.7]),
            "label": 1
        })
    
    return pd.DataFrame(all_profiles)

def preprocess_text(text):
    """Advanced text preprocessing for bio text"""
    if pd.isna(text):
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
    if pd.isna(text) or text == "":
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

def train_models():
    """Train multiple ML models and select the best one"""
    
    print("Creating synthetic dataset...")
    data = create_synthetic_dataset()
    print(f"Dataset created with {len(data)} profiles")
    print(f"Genuine profiles: {len(data[data['label'] == 0])}")
    print(f"Fake profiles: {len(data[data['label'] == 1])}")
    
    # Preprocess text
    print("Preprocessing text data...")
    data['bio_processed'] = data['bio'].apply(preprocess_text)
    
    # Extract text features
    print("Extracting text features...")
    text_features = data['bio_processed'].apply(extract_text_features)
    text_features_df = pd.DataFrame(text_features.tolist())
    
    # Prepare features
    X_text = data['bio_processed']
    X_numeric = data[['friends', 'posts', 'age_days']].copy()
    X_numeric['verified'] = data['verified'].astype(int)
    X_numeric['profile_pic'] = data['profile_pic'].astype(int)
    
    # Add extracted text features
    X_numeric = pd.concat([X_numeric, text_features_df], axis=1)
    
    y = data['label']
    
    # TF-IDF Vectorization
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    X_text_features = vectorizer.fit_transform(X_text)
    
    # Combine all features
    from scipy.sparse import hstack
    X_combined = hstack([X_text_features, X_numeric.values])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.toarray())
    X_test_scaled = scaler.transform(X_test.toarray())
    
    # Define models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    results = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"{name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Select best model based on F1 score
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with F1 score: {best_score:.3f}")
    
    # Save the best model and related objects
    model_data = {
        'model': best_model,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'feature_names': list(X_numeric.columns),
        'model_name': best_name,
        'results': results
    }
    
    # Ensure the directory exists
    os.makedirs('detection_app', exist_ok=True)
    
    with open('detection_app/ml_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to detection_app/ml_model.pkl")
    print(f"Model: {best_name}")
    print(f"Accuracy: {results[best_name]['accuracy']:.3f}")
    print(f"Precision: {results[best_name]['precision']:.3f}")
    print(f"Recall: {results[best_name]['recall']:.3f}")
    print(f"F1 Score: {results[best_name]['f1']:.3f}")
    
    return model_data, results

if __name__ == "__main__":
    print("=== Fake Profile Detection Model Training ===")
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model_data, results = train_models()
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll models trained and best model saved!")
