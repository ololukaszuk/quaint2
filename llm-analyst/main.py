#!/usr/bin/env python3
"""
LLM Market Analyst Service v2.0

Uses DeepSeek (or other Ollama models) to provide AI-powered market predictions
based on candle data and enhanced market-analyzer output.

KEY FEATURES:
- Configurable analysis interval (default: every 5 closed 1m candles)
- Predictions for +1hr and +4hrs from prediction moment
- Self-assessment from past predictions
- All 5 pivot methods (Traditional, Fibonacci, Camarilla, Woodie, DeMark)
- Strict output format enforcement

The LLM understands:
- It runs after every X closed 1m candles (configurable)
- It predicts price at +1hr and +4hrs from NOW
- Short-term predictions where 0.2-2% moves are significant
- Recent price action is critical for timing
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from loguru import logger

from config import Config
from database import Database
from ollama_client import OllamaClient, LLMResponse
from prompt_builder import build_analysis_prompt, build_system_prompt
from response_parser import parse_llm_response, ParsedAnalysis


# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)


class LLMAnalystService:
    """Main service that orchestrates LLM market analysis with enhanced data."""
    
    def __init__(self):
        self.config = Config()
        self.db: Optional[Database] = None
        self.ollama: Optional[OllamaClient] = None
        self.running = False
        
        # Track candles for interval triggering
        self.last_analysis_candle_time: Optional[datetime] = None
        self.candles_since_analysis: int = 0

        # Track active prediction state
        self.active_prediction: Optional[dict] = None
        
    async def start(self):
        """Initialize and start the service."""
        logger.info("=" * 70)
        logger.info("ðŸ¤– LLM MARKET ANALYST SERVICE v2.0 STARTING")
        logger.info("=" * 70)
        logger.info(f"Model: {self.config.ollama_model}")
        logger.info(f"Ollama URL: {self.config.ollama_base_url}")
        logger.info(f"Analysis interval: Every {self.config.analysis_interval_candles} closed 1m candles")
        logger.info(f"Predictions for: +1hr and +4hrs from prediction time")
        logger.info("Features: All 5 pivot methods, SMC, Signal Factors, Self-Assessment")
        logger.info("")
        
        # Connect to database
        self.db = Database(self.config)
        await self.db.connect()
        
        # Initialize Ollama client
        self.ollama = OllamaClient(self.config)
        
        # Check Ollama connectivity
        if not await self.ollama.health_check():
            logger.error(f"Cannot connect to Ollama at {self.config.ollama_base_url}")
            logger.error("Please ensure Ollama is running and accessible")
            return
        
        # List available models
        models = await self.ollama.list_models()
        logger.info(f"Available models: {', '.join(models) if models else 'unknown'}")
        
        # Check if configured model is available
        model_base = self.config.ollama_model.split(':')[0]
        available_bases = [m.split(':')[0] for m in models]
        model_available = self.config.ollama_model in models or model_base in available_bases
        
        if not model_available:
            logger.warning(f"Model {self.config.ollama_model} may not be available")
        
        # Get initial candle time
        self.last_analysis_candle_time = await self.db.get_latest_candle_time()
        if self.last_analysis_candle_time:
            logger.info(f"Last 1m candle OPEN time in DB: {self.last_analysis_candle_time}")
        else:
            logger.warning("No candles found in database yet")
        
        # Run initial analysis
        logger.info("Running initial analysis...")
        await self.run_analysis()
        
        self.running = True
        logger.info(f"Polling every {self.config.poll_interval_seconds}s for new candles")
        
        # Main loop
        while self.running:
            try:
                await self.check_for_trigger()
                await asyncio.sleep(self.config.poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(30)
    
    async def check_for_trigger(self):
        """
        Smart trigger: check prediction state and market requests.
        
        Only run new analysis when:
        1. Market-analyzer explicitly requests (pending llm_requests)
        2. Active prediction invalidated (price crossed invalidation level)
        3. Active prediction fulfilled (target reached)
        4. Prediction stale (4h+ with 2%+ move)
        """
        if not self.db or not self.db.pool:
            return
        
        try:
            # Priority 1: Check for market-analyzer request
            request = await self.check_pending_request()
            if request:
                logger.info(f"ðŸ“¥ Market request: {request['trigger_reason']}")
                await self.process_request(request['id'])
                return
            
            # Load active prediction if not cached
            if not self.active_prediction:
                self.active_prediction = await self.load_active_prediction()
            
            # If no active prediction, wait for market request
            if not self.active_prediction:
                return
            
            # Get current price
            current_price = await self.get_current_price()
            if not current_price:
                return
            
            # Check invalidation
            if (invalidation := self.check_invalidation(current_price)):
                logger.info(f"ðŸš¨ INVALIDATED: {invalidation}")
                await self.run_new_analysis(f"invalidated:{invalidation}")
                return
            
            # Check fulfillment
            fulfillment = await self.check_fulfillment(current_price)
            if fulfillment:
                logger.info(f"âœ… TARGET HIT: {fulfillment}")
                await self.run_new_analysis(f"fulfilled:{fulfillment}")
                return
            
            # Check staleness
            if (staleness := self.check_staleness(current_price)):
                logger.info(f"â° STALE: {staleness}")
                await self.run_new_analysis(f"stale:{staleness}")
                return
                
        except Exception as e:
            logger.error(f"Error in check_for_trigger: {e}")
            import traceback
            traceback.print_exc()
            
    async def load_active_prediction(self) -> Optional[dict]:
        """Load most recent prediction from database."""
        if not self.db or not self.db.pool:
            return None
        
        try:
            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT 
                        id, analysis_time, price, 
                        prediction_direction, predicted_price_1h, predicted_price_4h,
                        invalidation_level, critical_support, critical_resistance
                    FROM llm_analysis
                    ORDER BY analysis_time DESC
                    LIMIT 1
                    """
                )
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error loading prediction: {e}")
            return None

    async def get_current_price(self) -> Optional[float]:
        """Get latest BTC price."""
        if not self.db or not self.db.pool:
            return None
        
        try:
            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT close FROM candles_1m ORDER BY time DESC LIMIT 1"
                )
                return float(row['close']) if row else None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None

    async def check_pending_request(self) -> Optional[dict]:
        """
        Check for market-analyzer request.
        Only processes LATEST request, marks older ones as superseded.
        """
        if not self.db or not self.db.pool:
            return None
        
        try:
            async with self.db.pool.acquire() as conn:
                # Get the LATEST pending request
                row = await conn.fetchrow(
                    """
                    SELECT id, trigger_reason, request_time
                    FROM llm_requests
                    WHERE status = 'pending'
                    ORDER BY request_time DESC
                    LIMIT 1
                    """
                )
                
                if not row:
                    return None
                
                latest_id = row['id']
                
                # Mark ALL other pending requests as superseded
                result = await conn.execute(
                    """
                    UPDATE llm_requests 
                    SET status = 'superseded', processed_at = NOW()
                    WHERE status = 'pending' AND id != $1
                    """,
                    latest_id
                )
                
                # Parse "UPDATE 5" -> 5
                superseded_count = int(result.split()[-1]) if result else 0
                
                if superseded_count > 0:
                    logger.info(f"ðŸ—‘ï¸  Marked {superseded_count} old requests as superseded")
                
                return dict(row)
        except Exception as e:
            logger.error(f"Error checking requests: {e}")
            return None

    def check_invalidation(self, current_price: float) -> Optional[str]:
        """Check if price crossed invalidation level."""
        if not self.active_prediction or not self.active_prediction.get('invalidation_level'):
            return None
        
        inv_level = float(self.active_prediction['invalidation_level'])
        direction = self.active_prediction['prediction_direction']
        
        if direction == "BULLISH" and current_price < inv_level:
            pct = ((inv_level - current_price) / current_price) * 100
            return f"price ${current_price:,.0f} < invalidation ${inv_level:,.0f} (-{pct:.2f}%)"
        
        if direction == "BEARISH" and current_price > inv_level:
            pct = ((current_price - inv_level) / current_price) * 100
            return f"price ${current_price:,.0f} > invalidation ${inv_level:,.0f} (+{pct:.2f}%)"
        
        return None

    async def check_fulfillment(self, current_price: float) -> Optional[str]:
        """
        Check if target reached by querying actual price history.
        
        Queries database to see if price ever reached the target during
        the predicted time window, not just current price.
        """
        if not self.active_prediction or not self.db or not self.db.pool:
            return None
        
        pred_time = self.active_prediction['analysis_time']
        direction = self.active_prediction['prediction_direction']
        target_1h = self.active_prediction.get('predicted_price_1h')
        target_4h = self.active_prediction.get('predicted_price_4h')
        
        # Calculate time elapsed since prediction
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - pred_time).total_seconds() / 3600
        minutes_elapsed = hours_elapsed * 60
        
        # Check 4h target (window: 3.5 to 4.5 hours)
        if target_4h and 3.5 <= hours_elapsed <= 4.5:
            target_4h = float(target_4h)
            
            try:
                async with self.db.pool.acquire() as conn:
                    if direction == "BULLISH":
                        # Check if price ever went above target
                        max_price = await conn.fetchval(
                            """
                            SELECT MAX(high) 
                            FROM candles_1m 
                            WHERE time >= $1 AND time <= $2
                            """,
                            pred_time,
                            now
                        )
                        
                        if max_price and max_price >= target_4h:
                            return f"4h target ${target_4h:,.0f} reached after {hours_elapsed:.1f}h (peak: ${max_price:,.0f})"
                    
                    else:  # BEARISH
                        # Check if price ever went below target
                        min_price = await conn.fetchval(
                            """
                            SELECT MIN(low) 
                            FROM candles_1m 
                            WHERE time >= $1 AND time <= $2
                            """,
                            pred_time,
                            now
                        )
                        
                        if min_price and min_price <= target_4h:
                            return f"4h target ${target_4h:,.0f} reached after {hours_elapsed:.1f}h (low: ${min_price:,.0f})"
            
            except Exception as e:
                logger.error(f"Error checking 4h target: {e}")
        
        # Check 1h target (window: 50 to 70 minutes)
        if target_1h and 50 <= minutes_elapsed <= 70:
            target_1h = float(target_1h)
            
            # Calculate 1 hour timestamp
            one_hour_ago = pred_time + timedelta(hours=1)
            
            try:
                async with self.db.pool.acquire() as conn:
                    if direction == "BULLISH":
                        # Check if price went above target in first hour
                        max_price = await conn.fetchval(
                            """
                            SELECT MAX(high) 
                            FROM candles_1m 
                            WHERE time >= $1 AND time <= $2
                            """,
                            pred_time,
                            one_hour_ago + timedelta(minutes=10)  # 10 min buffer
                        )
                        
                        if max_price and max_price >= target_1h:
                            return f"1h target ${target_1h:,.0f} reached after {minutes_elapsed:.0f}m (peak: ${max_price:,.0f})"
                    
                    else:  # BEARISH
                        # Check if price went below target in first hour
                        min_price = await conn.fetchval(
                            """
                            SELECT MIN(low) 
                            FROM candles_1m 
                            WHERE time >= $1 AND time <= $2
                            """,
                            pred_time,
                            one_hour_ago + timedelta(minutes=10)  # 10 min buffer
                        )
                        
                        if min_price and min_price <= target_1h:
                            return f"1h target ${target_1h:,.0f} reached after {minutes_elapsed:.0f}m (low: ${min_price:,.0f})"
            
            except Exception as e:
                logger.error(f"Error checking 1h target: {e}")
        
        return None

    def check_staleness(self, current_price: float) -> Optional[str]:
        """Check if prediction stale (4h+ with 2%+ move)."""
        if not self.active_prediction:
            return None
        
        pred_time = self.active_prediction['analysis_time']
        pred_price = float(self.active_prediction['price'])
        
        hours_elapsed = (datetime.now(timezone.utc) - pred_time).total_seconds() / 3600
        
        if hours_elapsed < 4:
            return None
        
        move_pct = abs((current_price - pred_price) / pred_price) * 100
        
        if move_pct > 2.0:
            return f"{hours_elapsed:.1f}h, {move_pct:.2f}% move from ${pred_price:,.0f}"
        
        return None

    async def process_request(self, request_id: int):
        """Process market-analyzer request and clean up old entries."""
        try:
            async with self.db.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE llm_requests SET status = 'processing' WHERE id = $1",
                    request_id
                )
            
            await self.run_analysis()
            
            async with self.db.pool.acquire() as conn:
                # Mark as completed
                await conn.execute(
                    """
                    UPDATE llm_requests 
                    SET status = 'completed', processed_at = NOW() 
                    WHERE id = $1
                    """,
                    request_id
                )
                
                # Optional: Clean up old completed/superseded requests (keep last 100)
                await conn.execute(
                    """
                    DELETE FROM llm_requests
                    WHERE id NOT IN (
                        SELECT id FROM llm_requests
                        ORDER BY request_time DESC
                        LIMIT 100
                    )
                    AND status IN ('completed', 'superseded', 'failed')
                    """
                )
            
            self.active_prediction = await self.load_active_prediction()
            logger.info(f"âœ… Request #{request_id} completed")
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            try:
                async with self.db.pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE llm_requests SET status = 'failed', error_message = $2 WHERE id = $1",
                        request_id, str(e)
                    )
            except:
                pass

    async def run_new_analysis(self, reason: str):
        """Run new analysis with reason."""
        logger.info(f"ðŸ”„ New analysis: {reason}")
        await self.run_analysis()
        self.active_prediction = await self.load_active_prediction()
    
    async def run_analysis(self):
        """Run full LLM analysis with enhanced data and self-assessment."""
        if not self.ollama or not self.db:
            return
        
        start_time = time.time()
        
        try:
            # Gather data
            logger.debug("Fetching market data (enhanced with all pivots)...")
            
            candles_1h = await self.db.get_candles_1h(limit=self.config.candles_1h_lookback)
            candles_15m = await self.db.get_candles_15m(limit=self.config.candles_15m_lookback)
            market_analysis = await self.db.get_latest_market_analysis()
            signal_history = await self.db.get_signal_history(limit=self.config.signal_history_count)
            
            # Get past predictions for self-assessment
            past_predictions = await self.db.get_past_predictions_with_outcomes(limit=10)
            
            logger.debug(f"Data: {len(candles_1h)} 1H candles, {len(candles_15m)} 15M candles")
            logger.debug(f"Signal history: {len(signal_history)} | Past predictions with outcomes: {len(past_predictions)}")
            
            if market_analysis:
                # Log pivot methods availability
                pivot_methods = []
                if market_analysis.get('pivot_daily'):
                    pivot_methods.append('Traditional')
                if market_analysis.get('pivot_r1_fibonacci'):
                    pivot_methods.append('Fibonacci')
                if market_analysis.get('pivot_r3_camarilla'):
                    pivot_methods.append('Camarilla')
                if market_analysis.get('pivot_woodie'):
                    pivot_methods.append('Woodie')
                if market_analysis.get('pivot_demark'):
                    pivot_methods.append('DeMark')
                logger.debug(f"Pivot methods available: {', '.join(pivot_methods) if pivot_methods else 'none'}")
            
            # Get current price
            if candles_1h:
                current_price = float(candles_1h[-1]['close'])
            elif market_analysis:
                current_price = float(market_analysis['price'])
            else:
                logger.warning("No price data available")
                return
            
            # Build system prompt with interval context
            system_prompt = build_system_prompt(self.config.analysis_interval_candles)
            
            # Build prompt with enhanced data and past predictions
            prompt = build_analysis_prompt(
                candles_1h=candles_1h,
                candles_15m=candles_15m,
                market_analysis=market_analysis,
                signal_history=signal_history,
                current_price=current_price,
                past_predictions=past_predictions,
                analysis_interval_candles=self.config.analysis_interval_candles,
            )
            
            logger.debug(f"Prompt built: {len(prompt)} chars")
            
            # Query LLM
            logger.info(f"ðŸ§  Querying {self.config.ollama_model}...")
            
            response = await self.ollama.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=4096,
            )
            
            if not response:
                logger.error("No response from LLM")
                return
            
            # Parse response
            parsed = parse_llm_response(response.content)
            
            # Validate parsed response
            parsed = self.validate_and_fix_response(parsed, current_price)
            
            elapsed = time.time() - start_time
            
            # Log the analysis (toggle between brief/detailed)
            if self.config.detailed_logging:
                self.log_analysis(parsed, response, current_price, elapsed, market_analysis, past_predictions)
            else:
                self.log_analysis_brief(parsed, response, current_price, elapsed)

            # Save to database
            await self.save_analysis_to_db(
                parsed=parsed,
                response=response,
                current_price=current_price,
                elapsed=elapsed,
                market_analysis=market_analysis,
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def validate_and_fix_response(self, parsed: ParsedAnalysis, current_price: float) -> ParsedAnalysis:
        """
        Validate and fix common issues with LLM responses:
        - Invalidation level direction (bullish should have lower invalidation)
        - Missing required fields
        - Contradictory outputs
        """
        # Fix invalidation level logic
        if parsed.invalidation_level and parsed.direction != "NEUTRAL":
            if parsed.direction == "BULLISH":
                # Bullish prediction should have invalidation BELOW current price
                if parsed.invalidation_level > current_price:
                    logger.warning(f"âš ï¸ Fixing invalidation: BULLISH but invalidation ${parsed.invalidation_level:,.0f} > price ${current_price:,.0f}")
                    # Use critical support or calculate from price
                    if parsed.critical_support and parsed.critical_support < current_price:
                        parsed.invalidation_level = parsed.critical_support * 0.998
                    else:
                        parsed.invalidation_level = current_price * 0.99
                    logger.info(f"   Fixed to ${parsed.invalidation_level:,.0f}")
                    
            elif parsed.direction == "BEARISH":
                # Bearish prediction should have invalidation ABOVE current price
                if parsed.invalidation_level < current_price:
                    if self.config.detailed_logging:
                        logger.warning(f"âš ï¸ Fixing invalidation: BEARISH but invalidation ${parsed.invalidation_level:,.0f} < price ${current_price:,.0f}")
                    if parsed.critical_resistance and parsed.critical_resistance > current_price:
                        parsed.invalidation_level = parsed.critical_resistance * 1.002
                    else:
                        parsed.invalidation_level = current_price * 1.01
                    logger.info(f"   Fixed to ${parsed.invalidation_level:,.0f}")
        
        # Ensure price targets make sense
        if parsed.price_1h:
            if parsed.direction == "BULLISH" and parsed.price_1h < current_price:
                logger.warning(f"âš ï¸ BULLISH but 1h target ${parsed.price_1h:,.0f} < current ${current_price:,.0f}")
            elif parsed.direction == "BEARISH" and parsed.price_1h > current_price:
                logger.warning(f"âš ï¸ BEARISH but 1h target ${parsed.price_1h:,.0f} > current ${current_price:,.0f}")
        
        # Fill in missing critical levels with defaults
        if not parsed.critical_support:
            parsed.critical_support = current_price * 0.99
        if not parsed.critical_resistance:
            parsed.critical_resistance = current_price * 1.01
        
        return parsed
    
    def log_analysis(
        self,
        parsed: ParsedAnalysis,
        response: LLMResponse,
        current_price: float,
        elapsed: float,
        market_analysis: Optional[Dict[str, Any]],
        past_predictions: Optional[List[Dict[str, Any]]] = None,
    ):
        """Log the analysis with rich formatting."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"ðŸ¤– LLM ANALYSIS COMPLETE - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("=" * 80)
        logger.info(f"Model: {response.model} | Time: {elapsed:.1f}s | Tokens: {response.eval_count} | Speed: {response.tokens_per_second:.1f} t/s")
        logger.info("Analysis triggered by market-analyzer via llm_requests table")
        logger.info("")
        
        # Show past accuracy if available
        if past_predictions:
            correct_count = sum(1 for p in past_predictions if p.get('direction_correct_1h'))
            total_count = len(past_predictions)
            if total_count > 0:
                accuracy = correct_count / total_count * 100
                acc_emoji = "âœ…" if accuracy >= 60 else "âš ï¸" if accuracy >= 50 else "âŒ"
                logger.info(f"ðŸ“Š PAST ACCURACY (1H direction): {acc_emoji} {correct_count}/{total_count} ({accuracy:.0f}%)")
                logger.info("")
        
        # Market context from analyzer
        if market_analysis:
            signal_type = market_analysis.get('signal_type', 'N/A')
            signal_dir = market_analysis.get('signal_direction', 'N/A')
            smc_bias = market_analysis.get('smc_bias', 'N/A')
            
            dir_emoji = "ðŸŸ¢" if signal_dir == "LONG" else "ðŸ”´" if signal_dir == "SHORT" else "ðŸŸ¡"
            logger.info(f"ðŸ“ˆ MARKET ANALYZER SIGNAL: {dir_emoji} {signal_type} ({signal_dir})")
            logger.info(f"   SMC Bias: {smc_bias}")
            
            warnings = market_analysis.get('warnings')
            if warnings:
                if isinstance(warnings, list) and len(warnings) > 0:
                    logger.info(f"   âš ï¸ Warnings: {len(warnings)} active")
                    for msg in warnings[:3]:
                        if isinstance(msg, str):
                            logger.info(f"      â€¢ {msg[:60]}")
            
            logger.info("")
        
        # Direction banner
        if parsed.direction == "BULLISH":
            logger.info("ðŸŸ¢" * 20)
            logger.info(f"ðŸš€ LLM PREDICTION: BULLISH")
            logger.info(f"   Confidence: {parsed.confidence}")
            logger.info("ðŸŸ¢" * 20)
        elif parsed.direction == "BEARISH":
            logger.info("ðŸ”´" * 20)
            logger.info(f"ðŸ“‰ LLM PREDICTION: BEARISH")
            logger.info(f"   Confidence: {parsed.confidence}")
            logger.info("ðŸ”´" * 20)
        else:
            logger.info("ðŸŸ¡" * 20)
            logger.info(f"â¸ï¸  LLM PREDICTION: NEUTRAL")
            logger.info(f"   Confidence: {parsed.confidence}")
            logger.info("ðŸŸ¡" * 20)
        
        logger.info("")
        
        # Price targets
        logger.info("ðŸ“Š PRICE TARGETS")
        logger.info("-" * 60)
        logger.info(f"  Current Price:  ${current_price:,.2f}")
        
        if parsed.price_1h:
            diff_1h = parsed.price_1h - current_price
            diff_pct_1h = diff_1h / current_price * 100
            dir_emoji = "ðŸ“ˆ" if diff_pct_1h > 0 else "ðŸ“‰"
            logger.info(f"  Expected (1H):  ${parsed.price_1h:,.0f} ({diff_pct_1h:+.2f}%) {dir_emoji}")
        else:
            logger.info(f"  Expected (1H):  Not specified âš ï¸")
        
        if parsed.price_4h:
            diff_4h = parsed.price_4h - current_price
            diff_pct_4h = diff_4h / current_price * 100
            dir_emoji = "ðŸ“ˆ" if diff_pct_4h > 0 else "ðŸ“‰"
            logger.info(f"  Expected (4H):  ${parsed.price_4h:,.0f} ({diff_pct_4h:+.2f}%) {dir_emoji}")
        else:
            logger.info(f"  Expected (4H):  Not specified âš ï¸")
        
        if parsed.invalidation_level:
            inv_pct = (parsed.invalidation_level - current_price) / current_price * 100
            logger.info(f"  Invalidation:   ${parsed.invalidation_level:,.0f} ({inv_pct:+.2f}%)")
        else:
            logger.info(f"  Invalidation:   Not specified âš ï¸")
        
        logger.info("")
        
        # Key levels
        logger.info("ðŸŽ¯ KEY LEVELS")
        logger.info("-" * 60)
        if parsed.critical_support:
            dist = (current_price - parsed.critical_support) / current_price * 100
            logger.info(f"  Critical Support:    ${parsed.critical_support:,.0f} ({dist:+.2f}% from price)")
        if parsed.critical_resistance:
            dist = (parsed.critical_resistance - current_price) / current_price * 100
            logger.info(f"  Critical Resistance: ${parsed.critical_resistance:,.0f} (+{dist:.2f}% from price)")
        
        logger.info("")
        
        # Reasoning
        logger.info("ðŸ’­ REASONING")
        logger.info("-" * 60)
        if parsed.reasoning:
            # Word wrap reasoning
            words = parsed.reasoning.split()
            lines = []
            current_line = "  "
            for word in words:
                if len(current_line) + len(word) + 1 > 75:
                    lines.append(current_line)
                    current_line = "  " + word
                else:
                    current_line += " " + word if current_line != "  " else word
            if current_line.strip():
                lines.append(current_line)
            
            for line in lines:
                logger.info(line)
        else:
            logger.info("  No specific reasoning extracted âš ï¸")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("")

    def log_analysis_brief(
        self,
        parsed: ParsedAnalysis,
        response: LLMResponse,
        current_price: float,
        elapsed: float,
    ):
        """Brief one-line logging for production."""
        timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
        
        # Direction emoji
        if parsed.direction == "BULLISH":
            emoji = "ðŸŸ¢"
        elif parsed.direction == "BEARISH":
            emoji = "ðŸ”´"
        else:
            emoji = "âšª"
        
        # Show 1h target and change
        if parsed.price_1h:
            change_pct = (parsed.price_1h - current_price) / current_price * 100
            target_str = f"â†’ ${parsed.price_1h:,.0f} ({change_pct:+.2f}%)"
        else:
            target_str = "â†’ N/A"
        
        logger.info(
            f"{timestamp} | ðŸ¤– {emoji} {parsed.direction} {parsed.confidence} | "
            f"${current_price:,.0f} {target_str} | {elapsed:.1f}s"
        )

    async def save_analysis_to_db(
        self,
        parsed: ParsedAnalysis,
        response: LLMResponse,
        current_price: float,
        elapsed: float,
        market_analysis: Optional[Dict[str, Any]] = None,
    ):
        """Save LLM analysis to database with full market context."""
        if not self.db:
            return
        
        # Extract market context for storage
        market_context = None
        signal_factors_used = None
        smc_bias_at_analysis = None
        trends_at_analysis = None
        warnings_at_analysis = None
        
        if market_analysis:
            market_context = {
                'signal_type': market_analysis.get('signal_type'),
                'signal_direction': market_analysis.get('signal_direction'),
                'signal_confidence': market_analysis.get('signal_confidence'),
                'smc_bias': market_analysis.get('smc_bias'),
                'price_zone': market_analysis.get('price_zone') or market_analysis.get('smc_price_zone'),
                'action_recommendation': market_analysis.get('action_recommendation'),
                'nearest_support': float(market_analysis['nearest_support']) if market_analysis.get('nearest_support') else None,
                'nearest_resistance': float(market_analysis['nearest_resistance']) if market_analysis.get('nearest_resistance') else None,
            }
            
            sf = market_analysis.get('signal_factors')
            if sf:
                if isinstance(sf, str):
                    try:
                        sf = json.loads(sf)
                    except:
                        sf = None
                signal_factors_used = sf
            
            smc_bias_at_analysis = market_analysis.get('smc_bias')
            
            trends = market_analysis.get('trends')
            if trends:
                if isinstance(trends, str):
                    try:
                        trends = json.loads(trends)
                    except:
                        trends = None
                trends_at_analysis = trends
            
            warnings = market_analysis.get('warnings')
            if warnings:
                if isinstance(warnings, str):
                    try:
                        warnings = json.loads(warnings)
                    except:
                        warnings = None
                warnings_at_analysis = warnings
        
        # Build key levels string
        key_levels = ""
        if parsed.critical_support and parsed.critical_resistance:
            key_levels = f"S: ${parsed.critical_support:,.0f} | R: ${parsed.critical_resistance:,.0f}"
        
        # Save to database
        await self.db.save_llm_analysis(
            analysis_time=datetime.now(timezone.utc),
            price=current_price,
            prediction_direction=parsed.direction,
            prediction_confidence=parsed.confidence,
            predicted_price_1h=parsed.price_1h,
            predicted_price_4h=parsed.price_4h,
            key_levels=key_levels,
            reasoning=parsed.reasoning or "",
            full_response=response.content,
            model_name=response.model,
            response_time_seconds=elapsed,
            # Enhanced fields
            invalidation_level=parsed.invalidation_level,
            critical_support=parsed.critical_support,
            critical_resistance=parsed.critical_resistance,
            market_context=market_context,
            signal_factors_used=signal_factors_used,
            smc_bias_at_analysis=smc_bias_at_analysis,
            trends_at_analysis=trends_at_analysis,
            warnings_at_analysis=warnings_at_analysis,
        )
        
        logger.debug("âœ… LLM analysis saved to database")
    
    async def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down LLM Analyst...")
        self.running = False
        if self.db:
            await self.db.disconnect()
        logger.info("Shutdown complete")


async def main():
    service = LLMAnalystService()
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))
    
    try:
        await service.start()
    except KeyboardInterrupt:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())