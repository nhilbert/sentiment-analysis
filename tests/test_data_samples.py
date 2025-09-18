#!/usr/bin/env python3
"""
Test data samples and fixtures for testing
"""

import pandas as pd

def create_sample_feedback_data():
    """Create sample German feedback data for testing"""
    return pd.DataFrame({
        'date': [
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'
        ],
        'feedback': [
            'Der Kundensupport war ausgezeichnet und sehr hilfreich!',
            'Die App-Oberfläche ist schlecht und unübersichtlich.',
            'Login-Probleme seit dem letzten Update.',
            'Schnelle Lieferung, bin sehr zufrieden.',
            'Rechnungs-PDF lässt sich nicht herunterladen.',
            'Support hat schnell geantwortet, super Service!',
            'Synchronisation mit Kalender funktioniert nicht.',
            'Neue Dashboard-Layout ist praktisch.',
            'Zu viele Schritte beim Stornieren.',
            'Alles funktioniert einwandfrei, danke!'
        ]
    })

def create_mixed_language_data():
    """Create mixed German/English feedback data"""
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'feedback': [
            'Der Service war excellent!',
            'Login form rejects valid credentials.',
            'Shipping took too long, nicht zufrieden.',
            'Great performance after update, sehr gut!'
        ]
    })

def create_edge_case_data():
    """Create edge case data for testing robustness"""
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'feedback': [
            '',  # Empty feedback
            'a',  # Very short feedback
            'This is a very long feedback message that contains multiple sentences and should test how the system handles longer text inputs with various punctuation marks, numbers like 123, and special characters like @#$%.',
            None  # Null feedback
        ]
    })

# Expected sentiment results for validation
EXPECTED_SENTIMENTS = {
    'Der Kundensupport war ausgezeichnet und sehr hilfreich!': 'positive',
    'Die App-Oberfläche ist schlecht und unübersichtlich.': 'negative',
    'Login-Probleme seit dem letzten Update.': 'negative',
    'Schnelle Lieferung, bin sehr zufrieden.': 'positive',
    'Alles funktioniert einwandfrei, danke!': 'positive'
}