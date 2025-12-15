//! Ensemble Module
//!
//! Combines predictions from Mamba and LightGBM models
//! using configurable weights.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::features::NUM_HORIZONS;
use crate::models::ModelPrediction;

/// Horizons in minutes for predictions
pub const HORIZON_MINUTES: [i32; NUM_HORIZONS] = [1, 2, 3, 4, 5];

/// Combined ensemble prediction result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    /// Timestamp when prediction was made
    pub prediction_time: DateTime<Utc>,

    /// Target times for each horizon
    pub target_times: Vec<DateTime<Utc>>,

    /// Final ensemble predictions (log returns)
    pub predictions: Vec<f64>,

    /// Confidence scores (0-1)
    pub confidences: Vec<f64>,

    /// Individual Mamba predictions
    pub mamba_predictions: Vec<f64>,

    /// Individual LightGBM predictions
    pub lgbm_predictions: Vec<f64>,

    /// Mamba weight used
    pub mamba_weight: f64,

    /// Total inference latency in milliseconds
    pub latency_ms: f64,

    /// Ensemble version ID
    pub ensemble_version_id: i32,
}

impl EnsemblePrediction {
    /// Create new ensemble prediction by combining model outputs
    pub fn combine(
        mamba_pred: &ModelPrediction,
        lgbm_pred: &ModelPrediction,
        mamba_weight: f64,
        ensemble_version_id: i32,
    ) -> Self {
        let now = Utc::now();
        let lgbm_weight = 1.0 - mamba_weight;

        // Calculate target times
        let target_times: Vec<DateTime<Utc>> = HORIZON_MINUTES
            .iter()
            .map(|&m| now + chrono::Duration::minutes(m as i64))
            .collect();

        // Combine predictions
        let predictions: Vec<f64> = mamba_pred
            .predictions
            .iter()
            .zip(lgbm_pred.predictions.iter())
            .map(|(&m, &l)| mamba_weight * m + lgbm_weight * l)
            .collect();

        // Calculate confidence based on agreement between models
        let confidences: Vec<f64> = mamba_pred
            .predictions
            .iter()
            .zip(lgbm_pred.predictions.iter())
            .map(|(&m, &l)| {
                let diff = (m - l).abs();
                let avg_magnitude = (m.abs() + l.abs()) / 2.0 + 1e-8;
                let agreement = 1.0 - (diff / avg_magnitude).min(1.0);
                agreement.max(0.0).min(1.0)
            })
            .collect();

        let latency_ms = mamba_pred.latency_ms + lgbm_pred.latency_ms;

        Self {
            prediction_time: now,
            target_times,
            predictions,
            confidences,
            mamba_predictions: mamba_pred.predictions.clone(),
            lgbm_predictions: lgbm_pred.predictions.clone(),
            mamba_weight,
            latency_ms,
            ensemble_version_id,
        }
    }

    /// Get predicted direction for a horizon
    pub fn predicted_direction(&self, horizon_idx: usize) -> i8 {
        if horizon_idx >= self.predictions.len() {
            return 0;
        }

        let pred = self.predictions[horizon_idx];
        if pred > 0.0 {
            1
        } else if pred < 0.0 {
            -1
        } else {
            0
        }
    }

    /// Get predicted price change percentage for a horizon
    pub fn predicted_change_pct(&self, horizon_idx: usize) -> f64 {
        if horizon_idx >= self.predictions.len() {
            return 0.0;
        }

        // Convert log return to percentage
        (self.predictions[horizon_idx].exp() - 1.0) * 100.0
    }

    /// Check if prediction should be acted upon (high confidence)
    pub fn is_actionable(&self, horizon_idx: usize, min_confidence: f64) -> bool {
        if horizon_idx >= self.confidences.len() {
            return false;
        }

        self.confidences[horizon_idx] >= min_confidence
    }

    /// Serialize to JSON for storage
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Ensemble metrics for monitoring
#[derive(Clone, Debug, Default)]
pub struct EnsembleMetrics {
    /// Total predictions made
    pub total_predictions: u64,

    /// Average latency in milliseconds
    pub avg_latency_ms: f64,

    /// Average confidence across all predictions
    pub avg_confidence: f64,

    /// Count of predictions by direction
    pub direction_counts: [u64; 3], // [down, neutral, up]

    /// Running sum for average calculation
    latency_sum: f64,
    confidence_sum: f64,
}

impl EnsembleMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update metrics with a new prediction
    pub fn update(&mut self, prediction: &EnsemblePrediction) {
        self.total_predictions += 1;

        // Update latency average
        self.latency_sum += prediction.latency_ms;
        self.avg_latency_ms = self.latency_sum / self.total_predictions as f64;

        // Update confidence average (use first horizon)
        if let Some(&conf) = prediction.confidences.first() {
            self.confidence_sum += conf;
            self.avg_confidence = self.confidence_sum / self.total_predictions as f64;
        }

        // Update direction counts (use first horizon)
        let direction = prediction.predicted_direction(0);
        match direction {
            -1 => self.direction_counts[0] += 1,
            0 => self.direction_counts[1] += 1,
            1 => self.direction_counts[2] += 1,
            _ => {}
        }
    }

    /// Get direction distribution as percentages
    pub fn direction_distribution(&self) -> [f64; 3] {
        let total = self.total_predictions.max(1) as f64;
        [
            self.direction_counts[0] as f64 / total * 100.0,
            self.direction_counts[1] as f64 / total * 100.0,
            self.direction_counts[2] as f64 / total * 100.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_predictions() {
        let mamba = ModelPrediction {
            predictions: vec![0.001, 0.002, 0.003, 0.004, 0.005],
            latency_ms: 10.0,
        };

        let lgbm = ModelPrediction {
            predictions: vec![0.002, 0.003, 0.004, 0.005, 0.006],
            latency_ms: 5.0,
        };

        let ensemble = EnsemblePrediction::combine(&mamba, &lgbm, 0.5, 1);

        // Check combined predictions (average when weight = 0.5)
        assert!((ensemble.predictions[0] - 0.0015).abs() < 1e-10);
        assert!((ensemble.predictions[4] - 0.0055).abs() < 1e-10);

        // Check latency
        assert!((ensemble.latency_ms - 15.0).abs() < 1e-10);

        // Check confidences are between 0 and 1
        for conf in &ensemble.confidences {
            assert!(*conf >= 0.0 && *conf <= 1.0);
        }
    }

    #[test]
    fn test_predicted_direction() {
        let mamba = ModelPrediction {
            predictions: vec![0.001, -0.002, 0.0, 0.003, -0.001],
            latency_ms: 10.0,
        };

        let lgbm = ModelPrediction {
            predictions: vec![0.001, -0.002, 0.0, 0.003, -0.001],
            latency_ms: 5.0,
        };

        let ensemble = EnsemblePrediction::combine(&mamba, &lgbm, 0.5, 1);

        assert_eq!(ensemble.predicted_direction(0), 1); // Positive
        assert_eq!(ensemble.predicted_direction(1), -1); // Negative
        assert_eq!(ensemble.predicted_direction(2), 0); // Zero
    }

    #[test]
    fn test_metrics_update() {
        let mut metrics = EnsembleMetrics::new();

        let pred = EnsemblePrediction {
            prediction_time: Utc::now(),
            target_times: vec![Utc::now(); 5],
            predictions: vec![0.001; 5],
            confidences: vec![0.8; 5],
            mamba_predictions: vec![0.001; 5],
            lgbm_predictions: vec![0.001; 5],
            mamba_weight: 0.5,
            latency_ms: 10.0,
            ensemble_version_id: 1,
        };

        metrics.update(&pred);

        assert_eq!(metrics.total_predictions, 1);
        assert!((metrics.avg_latency_ms - 10.0).abs() < 1e-10);
        assert!((metrics.avg_confidence - 0.8).abs() < 1e-10);
        assert_eq!(metrics.direction_counts[2], 1); // One "up" prediction
    }
}
