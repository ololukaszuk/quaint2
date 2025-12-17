#!/usr/bin/env python3
"""
LLM Market Analyst Service

Uses DeepSeek (or other Ollama models) to provide AI-powered market commentary
based on candle data and enhanced market-analyzer output.

Now includes:
- Full utilization of enhanced market_analysis schema (pivots, SMC, signals, etc.)
- Enhanced logging with all analysis details
- Saving market context to database for analysis

Runs every 5 closed 1m candles (configurable).
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from loguru import logger

from config import Config
from database import Database
from ollama_client import OllamaClient, LLMResponse
from prompt_builder import build_analysis_prompt, SYSTEM_PROMPT
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
        
    async def start(self):
        """Initialize and start the service."""
        logger.info("=" * 70)
        logger.info("🤖 LLM MARKET ANALYST SERVICE STARTING (Enhanced)")
        logger.info("=" * 70)
        logger.info(f"Model: {self.config.ollama_model}")
        logger.info(f"Ollama URL: {self.config.ollama_base_url}")
        logger.info(f"Analysis interval: Every {self.config.analysis_interval_candles} candles")
        logger.info("Features: Enhanced market data, SMC, Pivots, Signal Factors")
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
            logger.info(f"Last candle in DB: {self.last_analysis_candle_time}")
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
        """Check if we should run analysis based on candle count."""
        if not self.last_analysis_candle_time:
            self.last_analysis_candle_time = await self.db.get_latest_candle_time()
            return
        
        # Count new candles since last analysis
        new_candles = await self.db.get_candle_count_since(self.last_analysis_candle_time)
        
        if new_candles >= self.config.analysis_interval_candles:
            logger.info(f"📊 Trigger: {new_candles} new candles since last analysis")
            
            # Update tracking
            self.last_analysis_candle_time = await self.db.get_latest_candle_time()
            
            # Run analysis
            await self.run_analysis()
    
    async def run_analysis(self):
        """Run full LLM analysis with enhanced data and detailed logging."""
        if not self.ollama or not self.db:
            return
        
        start_time = time.time()
        
        try:
            # Gather data
            logger.debug("Fetching market data (enhanced)...")
            
            candles_1h = await self.db.get_candles_1h(limit=self.config.candles_1h_lookback)
            candles_15m = await self.db.get_candles_15m(limit=self.config.candles_15m_lookback)
            market_analysis = await self.db.get_latest_market_analysis()
            signal_history = await self.db.get_signal_history(limit=self.config.signal_history_count)
            
            logger.debug(f"Data: {len(candles_1h)} 1H candles, {len(candles_15m)} 15M candles, {len(signal_history)} signals")
            
            if market_analysis:
                # Log enhanced data availability
                enhanced_fields = ['signal_factors', 'smc_order_blocks', 'pivot_confluence_zones', 'warnings']
                available = [f for f in enhanced_fields if market_analysis.get(f)]
                logger.debug(f"Enhanced fields available: {', '.join(available) if available else 'none'}")
            
            # Get current price
            if candles_1h:
                current_price = float(candles_1h[-1]['close'])
            elif market_analysis:
                current_price = float(market_analysis['price'])
            else:
                logger.warning("No price data available")
                return
            
            # Build prompt with enhanced data
            prompt = build_analysis_prompt(
                candles_1h=candles_1h,
                candles_15m=candles_15m,
                market_analysis=market_analysis,
                signal_history=signal_history,
                current_price=current_price,
            )
            
            logger.debug(f"Prompt built: {len(prompt)} chars")
            
            # Query LLM
            logger.info(f"🧠 Querying {self.config.ollama_model}...")
            
            response = await self.ollama.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=1024,
            )
            
            if not response:
                logger.error("No response from LLM")
                return
            
            # Parse response
            parsed = parse_llm_response(response.content)
            
            elapsed = time.time() - start_time
            
            # Log the analysis (enhanced)
            self.log_analysis(parsed, response, current_price, elapsed, market_analysis)
            
            # Save to database with enhanced context
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
    
    def log_analysis(
        self,
        parsed: ParsedAnalysis,
        response: LLMResponse,
        current_price: float,
        elapsed: float,
        market_analysis: Optional[Dict[str, Any]] = None
    ):
        """Log LLM analysis with enhanced market context."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🤖 LLM MARKET ANALYSIS - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("=" * 80)
        logger.info(f"Model: {response.model} | Response time: {elapsed:.1f}s | Tokens: {response.eval_count}")
        logger.info("")
        
        # Market context from enhanced data
        if market_analysis:
            logger.info("📈 MARKET CONTEXT (from market-analyzer):")
            logger.info("-" * 60)
            logger.info(f"  Signal: {market_analysis.get('signal_type', 'N/A')} ({market_analysis.get('signal_direction', 'N/A')})")
            logger.info(f"  Confidence: {market_analysis.get('signal_confidence', 0):.0f}%")
            
            if market_analysis.get('smc_bias'):
                logger.info(f"  SMC Bias: {market_analysis['smc_bias']}")
            
            if market_analysis.get('action_recommendation'):
                logger.info(f"  Action: {market_analysis['action_recommendation']}")
            
            # Log top signal factors
            signal_factors = market_analysis.get('signal_factors')
            if signal_factors:
                if isinstance(signal_factors, str):
                    try:
                        signal_factors = json.loads(signal_factors)
                    except:
                        signal_factors = None
                
                if signal_factors and isinstance(signal_factors, list):
                    logger.info("")
                    logger.info("  Top Signal Factors:")
                    for factor in signal_factors[:5]:
                        weight = factor.get('weight', 0)
                        desc = factor.get('description', 'Unknown')
                        symbol = "🟢" if weight > 0 else "🔴" if weight < 0 else "⚪"
                        weight_pct = f"{weight*100:+.1f}%"
                        logger.info(f"    {symbol} {weight_pct:>7s} | {desc}")
            
            # Log warnings (stored as list of strings)
            warnings = market_analysis.get('warnings')
            if warnings:
                if isinstance(warnings, str):
                    try:
                        warnings = json.loads(warnings)
                    except:
                        warnings = None
                
                if warnings and isinstance(warnings, list):
                    logger.info("")
                    logger.info("  ⚠️ Active Warnings:")
                    for warning in warnings:
                        # Warnings are strings, not dicts
                        if isinstance(warning, str):
                            logger.info(f"    • {warning}")
                        elif isinstance(warning, dict):
                            msg = warning.get('message', warning.get('type', 'Unknown'))
                            logger.info(f"    • {msg}")
            
            logger.info("")
        
        # Direction banner
        if parsed.direction == "BULLISH":
            logger.info("🟢" * 20)
            logger.info(f"🚀 LLM PREDICTION: BULLISH")
            logger.info(f"   Confidence: {parsed.confidence}")
            logger.info("🟢" * 20)
        elif parsed.direction == "BEARISH":
            logger.info("🔴" * 20)
            logger.info(f"📉 LLM PREDICTION: BEARISH")
            logger.info(f"   Confidence: {parsed.confidence}")
            logger.info("🔴" * 20)
        else:
            logger.info("🟡" * 20)
            logger.info(f"⏸️  LLM PREDICTION: NEUTRAL")
            logger.info(f"   Confidence: {parsed.confidence}")
            logger.info("🟡" * 20)
        
        logger.info("")
        
        # Price targets
        logger.info("📊 PRICE TARGETS")
        logger.info("-" * 60)
        logger.info(f"  Current Price:  ${current_price:,.2f}")
        
        if parsed.price_1h:
            diff_1h = parsed.price_1h - current_price
            diff_pct_1h = diff_1h / current_price * 100
            logger.info(f"  Expected (1H):  ${parsed.price_1h:,.0f} ({diff_pct_1h:+.2f}%)")
        else:
            logger.info(f"  Expected (1H):  Not specified")
        
        if parsed.price_4h:
            diff_4h = parsed.price_4h - current_price
            diff_pct_4h = diff_4h / current_price * 100
            logger.info(f"  Expected (4H):  ${parsed.price_4h:,.0f} ({diff_pct_4h:+.2f}%)")
        else:
            logger.info(f"  Expected (4H):  Not specified")
        
        if parsed.invalidation_level:
            logger.info(f"  Invalidation:   ${parsed.invalidation_level:,.0f}")
        
        logger.info("")
        
        # Key levels
        logger.info("🎯 KEY LEVELS")
        logger.info("-" * 60)
        if parsed.critical_support:
            dist = (current_price - parsed.critical_support) / current_price * 100
            logger.info(f"  Critical Support:    ${parsed.critical_support:,.0f} ({dist:+.2f}% from price)")
        if parsed.critical_resistance:
            dist = (parsed.critical_resistance - current_price) / current_price * 100
            logger.info(f"  Critical Resistance: ${parsed.critical_resistance:,.0f} (+{dist:.2f}% from price)")
        
        logger.info("")
        
        # Reasoning
        logger.info("💭 REASONING")
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
            logger.info("  No specific reasoning extracted")
        
        logger.info("")
        
        # Full response (collapsed)
        logger.info("📝 FULL RESPONSE")
        logger.info("-" * 60)
        for line in response.content.split('\n'):
            if line.strip():
                logger.info(f"  {line[:100]}")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("")
    
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
            # Build compact market context
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
            
            # Signal factors
            sf = market_analysis.get('signal_factors')
            if sf:
                if isinstance(sf, str):
                    try:
                        sf = json.loads(sf)
                    except:
                        sf = None
                signal_factors_used = sf
            
            # SMC bias
            smc_bias_at_analysis = market_analysis.get('smc_bias')
            
            # Trends
            trends = market_analysis.get('trends')
            if trends:
                if isinstance(trends, str):
                    try:
                        trends = json.loads(trends)
                    except:
                        trends = None
                trends_at_analysis = trends
            
            # Warnings
            warnings = market_analysis.get('warnings')
            if warnings:
                if isinstance(warnings, str):
                    try:
                        warnings = json.loads(warnings)
                    except:
                        warnings = None
                warnings_at_analysis = warnings
        
        # Save to database
        await self.db.save_llm_analysis(
            analysis_time=datetime.now(timezone.utc),
            price=current_price,
            prediction_direction=parsed.direction,
            prediction_confidence=parsed.confidence,
            predicted_price_1h=parsed.price_1h,
            predicted_price_4h=parsed.price_4h,
            key_levels=f"S: ${parsed.critical_support:,.0f} | R: ${parsed.critical_resistance:,.0f}" if parsed.critical_support and parsed.critical_resistance else "",
            reasoning=parsed.reasoning,
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