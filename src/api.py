"""
Deployment API
FastAPI endpoint for serving RUL predictions with uncertainty estimation
Production-ready inference service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import logging
from datetime import datetime
import torch

from src.models import RULModels, get_torch_device

logger = logging.getLogger(__name__)


class SensorReading(BaseModel):
    """Single sensor reading"""
    engine_id: int
    timestamp: int
    sensor_values: List[float]
    operating_conditions: List[float]


class SensorWindow(BaseModel):
    """Window of readings (sequence)"""
    engine_id: int
    readings: List[List[float]]  # (timesteps, features)


class RULPredictionResponse(BaseModel):
    """RUL prediction response"""
    engine_id: int
    rul_point_estimate: float
    rul_lower_bound: float
    rul_upper_bound: float
    confidence: float
    confidence_level: str
    risk_level: str
    maintenance_recommendation: str
    timestamp: str
    model_version: str


class HealthReportResponse(BaseModel):
    """Detailed health report"""
    engine_id: int
    rul_point_estimate: float
    rul_confidence_interval: Dict
    health_score: float
    trend: str  # "DEGRADING", "STABLE", "IMPROVING"
    estimated_failure_date: str
    maintenance_schedule: str
    risk_assessment: Dict


class DeploymentAPI:
    """API for model serving"""
    
    def __init__(self, model_path: str, scaler_path: str, model_version: str = "1.0.0"):
        """
        Initialize API with trained model
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            model_version: Version string for tracking
        """
        self.device = get_torch_device()
        self.model = RULModels.load_model(model_path, map_location=self.device).to(self.device)
        self.model.eval()
        self.scaler = self._load_scaler(scaler_path)
        self.model_version = model_version
        logger.info(f"Model loaded: {model_path}")
    
    @staticmethod
    def _load_scaler(scaler_path: str):
        """Load scaler from joblib"""
        import joblib
        return joblib.load(scaler_path)
    
    def preprocess_input(self, readings: np.ndarray) -> np.ndarray:
        """
        Preprocess sensor readings
        
        Args:
            readings: (sequence_length, n_features)
            
        Returns:
            Normalized readings
        """
        # Normalize using training statistics
        n_steps, n_features = readings.shape
        readings_flat = readings.reshape(-1, n_features)
        readings_scaled = self.scaler.transform(readings_flat)
        return readings_scaled.reshape(n_steps, n_features)
    
    def predict_with_uncertainty(self, readings: np.ndarray, 
                                use_mc_dropout: bool = True,
                                n_iterations: int = 50) -> Dict:
        """
        Predict RUL with uncertainty estimation
        
        Args:
            readings: Sensor readings (sequence_length, n_features)
            use_mc_dropout: Use MC Dropout for uncertainty
            n_iterations: MC Dropout iterations
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        # Preprocess
        readings = self.preprocess_input(readings)
        readings = np.expand_dims(readings, 0)  # Add batch dimension
        readings_tensor = torch.from_numpy(readings.astype(np.float32)).to(self.device)
        
        if use_mc_dropout:
            # MC Dropout predictions
            predictions = []
            self.model.train()
            for _ in range(n_iterations):
                with torch.no_grad():
                    pred = self.model(readings_tensor).cpu().numpy().flatten()
                predictions.append(pred)
            predictions = np.array(predictions)
            self.model.eval()
            
            mean_pred = np.mean(predictions, axis=0)[0]
            std_pred = np.std(predictions, axis=0)[0]
        else:
            # Point prediction
            self.model.eval()
            with torch.no_grad():
                mean_pred = self.model(readings_tensor).cpu().numpy()[0]
            std_pred = 0.1 * mean_pred  # Approximate uncertainty
        
        return {
            'point_estimate': float(mean_pred),
            'std': float(std_pred),
            'lower_95': float(max(0, mean_pred - 1.96 * std_pred)),
            'upper_95': float(mean_pred + 1.96 * std_pred)
        }
    
    def predict_batch(self, batch_readings: List[np.ndarray]) -> List[Dict]:
        """
        Batch prediction
        
        Args:
            batch_readings: List of sensor windows
            
        Returns:
            List of predictions
        """
        results = []
        for readings in batch_readings:
            pred = self.predict_with_uncertainty(readings)
            results.append(pred)
        return results


def create_api(model_path: str, scaler_path: str) -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to scaler
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="NASA RUL Prediction API",
        description="Remaining Useful Life prediction for turbofan engines",
        version="1.0.0"
    )
    
    # Load model
    api = DeploymentAPI(model_path, scaler_path)
    
    @app.get("/health")
    def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_version": api.model_version
        }
    
    @app.post("/predict")
    def predict_rul(window: SensorWindow) -> RULPredictionResponse:
        """
        Predict RUL for engine
        
        Args:
            window: Sensor data window
            
        Returns:
            RUL prediction with confidence
        """
        try:
            readings = np.array(window.readings)
            prediction = api.predict_with_uncertainty(readings)
            
            # Determine confidence level
            confidence = prediction['std']
            if confidence < 5:
                confidence_level = "HIGH"
                confidence_score = 0.9
            elif confidence < 10:
                confidence_level = "MEDIUM"
                confidence_score = 0.7
            else:
                confidence_level = "LOW"
                confidence_score = 0.5
            
            # Risk assessment
            if prediction['point_estimate'] < 10:
                risk_level = "CRITICAL"
                recommendation = "IMMEDIATE maintenance required"
            elif prediction['point_estimate'] < 30:
                risk_level = "HIGH"
                recommendation = "Schedule maintenance soon"
            else:
                risk_level = "MEDIUM"
                recommendation = "Monitor and plan maintenance"
            
            return RULPredictionResponse(
                engine_id=window.engine_id,
                rul_point_estimate=prediction['point_estimate'],
                rul_lower_bound=prediction['lower_95'],
                rul_upper_bound=prediction['upper_95'],
                confidence=confidence_score,
                confidence_level=confidence_level,
                risk_level=risk_level,
                maintenance_recommendation=recommendation,
                timestamp=datetime.now().isoformat(),
                model_version=api.model_version
            )
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/health-report")
    def generate_health_report(window: SensorWindow) -> HealthReportResponse:
        """
        Generate detailed health report for engine
        
        Args:
            window: Sensor data window
            
        Returns:
            Comprehensive health assessment
        """
        try:
            readings = np.array(window.readings)
            prediction = api.predict_with_uncertainty(readings)
            
            # Calculate health score (simplified)
            # Higher RUL = better health
            health_score = min(1.0, prediction['point_estimate'] / 100)
            
            # Determine trend (simplified)
            if len(readings) > 10:
                early_mean = np.mean(readings[:5, :5])  # First 5 timesteps, first 5 sensors
                late_mean = np.mean(readings[-5:, :5])  # Last 5 timesteps, first 5 sensors
                if late_mean > early_mean:
                    trend = "DEGRADING"
                else:
                    trend = "STABLE"
            else:
                trend = "INSUFFICIENT_DATA"
            
            return HealthReportResponse(
                engine_id=window.engine_id,
                rul_point_estimate=prediction['point_estimate'],
                rul_confidence_interval={
                    'lower': prediction['lower_95'],
                    'upper': prediction['upper_95']
                },
                health_score=health_score,
                trend=trend,
                estimated_failure_date=f"{int(prediction['point_estimate'])} cycles",
                maintenance_schedule="Based on RUL prediction",
                risk_assessment={
                    'failure_probability': 0.3 if prediction['point_estimate'] > 50 else 0.7,
                    'recommendation': 'Monitor' if prediction['point_estimate'] > 30 else 'Schedule'
                }
            )
        
        except Exception as e:
            logger.error(f"Health report error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    # Example: uvicorn app:app --reload
    app = create_api("models/best_model.h5", "models/scaler.pkl")
    uvicorn.run(app, host="0.0.0.0", port=8000)
