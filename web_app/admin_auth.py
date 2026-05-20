"""
Admin Authentication & Background Learning Module
Restricts admin interface to authorized users
Implements silent background learning from transcription data
"""

import os
import json
import hashlib
from datetime import datetime
from functools import wraps
from flask import session, redirect, url_for, request, flash

# Admin credentials (hardcoded for security)
ADMIN_PASSWORD = hashlib.sha256(b"admin_key_2026").hexdigest()
ADMIN_USERNAME = "admin"

class AdminManager:
    """Manage admin access and background learning"""

    def __init__(self):
        # Use absolute paths based on web_app directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.learning_queue_file = os.path.join(base_dir, 'data', 'learning_queue.json')
        self.learning_log_file = os.path.join(base_dir, 'data', 'learning_log.json')
        self.learning_enabled = True
        self.load_queue()

    def load_queue(self):
        """Load pending transcriptions for background learning"""
        if os.path.exists(self.learning_queue_file):
            with open(self.learning_queue_file, 'r') as f:
                self.queue = json.load(f)
        else:
            self.queue = []

    def save_queue(self):
        """Save learning queue to disk"""
        os.makedirs(os.path.dirname(self.learning_queue_file), exist_ok=True)
        with open(self.learning_queue_file, 'w') as f:
            json.dump(self.queue, f)

    def add_to_learning_queue(self, audio_path, transcript, confidence):
        """Add transcription to background learning queue (silent)"""
        if not self.learning_enabled:
            return

        entry = {
            'timestamp': datetime.now().isoformat(),
            'audio_path': audio_path,
            'transcript': transcript,
            'confidence': confidence,
            'learned': False
        }
        self.queue.append(entry)
        self.save_queue()

        # Log without user knowing
        self._log_learning(entry)

    def _log_learning(self, entry):
        """Silent logging of learning data"""
        os.makedirs(os.path.dirname(self.learning_log_file), exist_ok=True)
        log = []
        if os.path.exists(self.learning_log_file):
            with open(self.learning_log_file, 'r') as f:
                log = json.load(f)

        log.append(entry)
        # Keep only last 1000 entries
        log = log[-1000:]

        with open(self.learning_log_file, 'w') as f:
            json.dump(log, f)

    def get_unlearned_samples(self, limit=10):
        """Get samples for background training"""
        unlearned = [s for s in self.queue if not s.get('learned', False)]
        return unlearned[:limit]

    def mark_learned(self, audio_path):
        """Mark sample as learned"""
        for item in self.queue:
            if item['audio_path'] == audio_path:
                item['learned'] = True
        self.save_queue()

    def get_statistics(self):
        """Get learning statistics (admin only)"""
        total = len(self.queue)
        learned = sum(1 for s in self.queue if s.get('learned', False))

        return {
            'total_samples': total,
            'learned': learned,
            'pending': total - learned,
            'learning_enabled': self.learning_enabled
        }

# Create global admin manager
admin_manager = AdminManager()

def login_required(f):
    """Decorator to require admin login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            flash('Admin access required', 'danger')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def check_admin_password(password):
    """Verify admin password"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == ADMIN_PASSWORD

def verify_admin_token(token):
    """Verify admin session token"""
    return session.get('admin_logged_in') and session.get('admin_token') == token
