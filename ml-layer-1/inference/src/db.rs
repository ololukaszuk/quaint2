//! Database Helpers Module
//!
//! Database operations for storing predictions and loading models.

use chrono::{DateTime, Utc};
use rust_decimal::prelude::*;
use std::sync::Arc;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::{debug, error, info};

use crate::ensemble::EnsemblePrediction;
use crate::features::NUM_HORIZONS;

/// Database errors
#[derive(Error, Debug)]
pub enum DbError {
    #[error("Database error: {0}")]
    Postgres(#[from] tokio_postgres::Error),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Query error: {0}")]
    Query(String),
}

/// Database operations for ML inference
pub struct InferenceDb {
    client: Arc<Client>,
}

impl InferenceDb {
    /// Create new database helper
    pub fn new(client: Arc<Client>) -> Self {
        Self { client }
    }

    /// Insert prediction into predictions table
    pub async fn insert_prediction(
        &self,
        prediction: &EnsemblePrediction,
    ) -> Result<(), DbError> {
        let model_name = "mamba_lgbm_ensemble";

        // Insert one row per horizon
        for (i, horizon) in [1, 2, 3, 4, 5].iter().enumerate() {
            if i >= prediction.predictions.len() {
                break;
            }

            let predicted_close = Decimal::from_f64(prediction.predictions[i])
                .unwrap_or(Decimal::ZERO);
            let confidence = Decimal::from_f64(prediction.confidences[i])
                .unwrap_or(Decimal::ZERO);
            let direction = prediction.predicted_direction(i) as i16;

            self.client
                .execute(
                    r#"
                    INSERT INTO predictions (
                        model_name, prediction_time, target_time, horizon,
                        predicted_close, predicted_direction, confidence,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    ON CONFLICT (prediction_time, model_name, horizon) 
                    DO UPDATE SET
                        predicted_close = EXCLUDED.predicted_close,
                        predicted_direction = EXCLUDED.predicted_direction,
                        confidence = EXCLUDED.confidence
                    "#,
                    &[
                        &model_name,
                        &prediction.prediction_time,
                        &prediction.target_times[i],
                        &(*horizon as i32),
                        &predicted_close,
                        &direction,
                        &confidence,
                    ],
                )
                .await?;
        }

        debug!(
            "Inserted prediction at {} with {} horizons",
            prediction.prediction_time,
            prediction.predictions.len()
        );

        Ok(())
    }

    /// Get active ensemble configuration
    pub async fn get_active_ensemble(&self) -> Result<Option<ActiveEnsemble>, DbError> {
        let row = self
            .client
            .query_opt(
                r#"
                SELECT 
                    id, mamba_version_id, lgbm_version_id, mamba_weight,
                    is_active, is_test, created_at
                FROM active_ensembles
                WHERE is_active = true
                LIMIT 1
                "#,
                &[],
            )
            .await?;

        Ok(row.map(|r| ActiveEnsemble {
            id: r.get("id"),
            mamba_version_id: r.get("mamba_version_id"),
            lgbm_version_id: r.get("lgbm_version_id"),
            mamba_weight: r
                .get::<_, Option<Decimal>>("mamba_weight")
                .and_then(|d| d.to_f64())
                .unwrap_or(0.5),
            is_active: r.get("is_active"),
            is_test: r.get::<_, Option<bool>>("is_test").unwrap_or(false),
            created_at: r.get("created_at"),
        }))
    }

    /// Update actual values for predictions when target time arrives
    pub async fn update_actuals(&self, target_time: DateTime<Utc>, actual_close: f64) -> Result<u64, DbError> {
        let actual = Decimal::from_f64(actual_close).unwrap_or(Decimal::ZERO);

        let result = self
            .client
            .execute(
                r#"
                UPDATE predictions
                SET 
                    actual_close = $1,
                    accuracy = CASE 
                        WHEN predicted_direction > 0 AND $1 > predicted_close THEN 1
                        WHEN predicted_direction < 0 AND $1 < predicted_close THEN 1
                        WHEN predicted_direction = 0 AND ABS($1 - predicted_close) / predicted_close < 0.001 THEN 1
                        ELSE 0
                    END,
                    rmse = SQRT(POWER(predicted_close - $1, 2)),
                    mape = CASE 
                        WHEN $1 > 0 THEN ABS(predicted_close - $1) / $1
                        ELSE NULL
                    END
                WHERE target_time = $2
                  AND actual_close IS NULL
                "#,
                &[&actual, &target_time],
            )
            .await?;

        if result > 0 {
            debug!("Updated {} predictions with actual_close at {}", result, target_time);
        }

        Ok(result)
    }

    /// Get recent prediction accuracy
    pub async fn get_recent_accuracy(&self, hours: i32) -> Result<AccuracyStats, DbError> {
        let row = self
            .client
            .query_one(
                r#"
                SELECT 
                    COUNT(*) as total,
                    COUNT(accuracy) as evaluated,
                    AVG(CASE WHEN accuracy IS NOT NULL THEN accuracy ELSE NULL END) as avg_accuracy,
                    AVG(rmse) as avg_rmse,
                    AVG(mape) as avg_mape
                FROM predictions
                WHERE prediction_time > NOW() - INTERVAL '1 hour' * $1
                "#,
                &[&hours],
            )
            .await?;

        Ok(AccuracyStats {
            total_predictions: row.get::<_, i64>("total") as u64,
            evaluated_predictions: row.get::<_, i64>("evaluated") as u64,
            avg_accuracy: row
                .get::<_, Option<Decimal>>("avg_accuracy")
                .and_then(|d| d.to_f64())
                .unwrap_or(0.0),
            avg_rmse: row
                .get::<_, Option<Decimal>>("avg_rmse")
                .and_then(|d| d.to_f64())
                .unwrap_or(0.0),
            avg_mape: row
                .get::<_, Option<Decimal>>("avg_mape")
                .and_then(|d| d.to_f64())
                .unwrap_or(0.0),
        })
    }

    /// Count pending predictions (waiting for actuals)
    pub async fn count_pending(&self) -> Result<i64, DbError> {
        let row = self
            .client
            .query_one(
                "SELECT COUNT(*) as count FROM predictions WHERE actual_close IS NULL",
                &[],
            )
            .await?;

        Ok(row.get("count"))
    }

    /// Log inference event
    pub async fn log_inference_event(
        &self,
        event_type: &str,
        details: &str,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                r#"
                INSERT INTO data_quality_logs (
                    event_type, source, error_message, resolved
                ) VALUES ($1, 'ml_inference', $2, true)
                "#,
                &[&event_type, &details],
            )
            .await?;

        Ok(())
    }
}

/// Active ensemble configuration from database
#[derive(Clone, Debug)]
pub struct ActiveEnsemble {
    pub id: i32,
    pub mamba_version_id: Option<i32>,
    pub lgbm_version_id: Option<i32>,
    pub mamba_weight: f64,
    pub is_active: bool,
    pub is_test: bool,
    pub created_at: DateTime<Utc>,
}

/// Accuracy statistics
#[derive(Clone, Debug, Default)]
pub struct AccuracyStats {
    pub total_predictions: u64,
    pub evaluated_predictions: u64,
    pub avg_accuracy: f64,
    pub avg_rmse: f64,
    pub avg_mape: f64,
}
