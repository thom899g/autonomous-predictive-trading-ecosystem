"""
Firebase Firestore client for state persistence and real-time data.
Implements singleton pattern with connection pooling and error recovery.
"""
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from config import config

logger = logging.getLogger(__name__)

class FirebaseClient:
    """Managed Firebase Firestore client with automatic reconnection."""
    
    _instance = None
    _client = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_firebase()
            self._initialized = True
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase app with error handling."""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase app initialized successfully")
            else:
                logger.info("Firebase app already initialized")
            
            self._client = firestore.client()
            logger.info("Firestore client created")
            
        except (ValueError, FirebaseError) as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self._client = None
    
    @property
    def client(self):
        """Lazy-loaded Firestore client."""
        if self._client is None:
            self._initialize_firebase()
        return self._client
    
    def save_prediction(self, 
                       model_id: str, 
                       prediction_data: Dict[str, Any],
                       collection_suffix: str = "predictions") -> Optional[str]:
        """
        Save prediction results to Firestore.
        
        Args:
            model_id: Unique identifier for the model
            prediction_data: Dictionary containing prediction results
            collection_suffix: Collection name suffix
            
        Returns:
            Document ID if successful, None otherwise
        """
        if not self.client:
            logger.error("Firestore client not available")
            return None
        
        try:
            collection_name = f"{config.FIRESTORE_COLLECTION_PREFIX}_{collection_suffix}"
            
            # Add metadata
            prediction_data.update({
                "timestamp": datetime.utcnow(),
                "model_id": model_id,
                "processed": False
            })
            
            doc_ref = self.client.collection(collection_name).document()
            doc_ref.set(prediction_data)
            
            logger.info(f"Prediction saved to {collection_name}/{doc_ref.id}")
            return doc_ref.id
            
        except FirebaseError as e:
            logger.error(f"Failed to save prediction: {e}")
            return None
    
    def get_latest_model_config(self, model_type: str = "lstm") -> Optional[Dict