"""
Response Parser for LLM Output

Extracts structured data from LLM responses.
"""

import re
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class ParsedAnalysis:
    """Structured analysis extracted from LLM response."""
    
    # Direction
    direction: str  # BULLISH, BEARISH, NEUTRAL
    confidence: str  # HIGH, MEDIUM, LOW
    
    # Price targets
    price_1h: Optional[float] = None
    price_4h: Optional[float] = None
    invalidation_level: Optional[float] = None
    
    # Key levels
    critical_support: Optional[float] = None
    critical_resistance: Optional[float] = None
    
    # Reasoning
    reasoning: str = ""
    
    # Raw response
    raw_response: str = ""


def extract_price(text: str, pattern: str) -> Optional[float]:
    """Extract price value matching pattern from text."""
    # Look for pattern followed by price
    match = re.search(rf'{pattern}[:\s]*\$?([\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        try:
            price_str = match.group(1).replace(',', '')
            return float(price_str)
        except ValueError:
            pass
    return None


def extract_direction(text: str) -> str:
    """Extract direction prediction from text."""
    text_upper = text.upper()
    
    # Look for explicit statements
    bullish_patterns = [
        r'DIRECTION[:\s]*BULLISH',
        r'PREDICTION[:\s]*BULLISH',
        r'EXPECT\s+(?:A\s+)?BULLISH',
        r'BIAS[:\s]*BULLISH',
        r'OUTLOOK[:\s]*BULLISH',
        r'\bBULLISH\b.*(?:LIKELY|EXPECTED|PROBABLE)',
        r'(?:LIKELY|EXPECTED|PROBABLE).*\bBULLISH\b',
        r'PRICE\s+(?:WILL|SHOULD|MAY)\s+(?:RISE|INCREASE|GO UP)',
        r'UPWARD\s+(?:MOVE|MOMENTUM|TREND)',
    ]
    
    bearish_patterns = [
        r'DIRECTION[:\s]*BEARISH',
        r'PREDICTION[:\s]*BEARISH',
        r'EXPECT\s+(?:A\s+)?BEARISH',
        r'BIAS[:\s]*BEARISH',
        r'OUTLOOK[:\s]*BEARISH',
        r'\bBEARISH\b.*(?:LIKELY|EXPECTED|PROBABLE)',
        r'(?:LIKELY|EXPECTED|PROBABLE).*\bBEARISH\b',
        r'PRICE\s+(?:WILL|SHOULD|MAY)\s+(?:FALL|DECREASE|DROP|GO DOWN)',
        r'DOWNWARD\s+(?:MOVE|MOMENTUM|TREND)',
    ]
    
    neutral_patterns = [
        r'DIRECTION[:\s]*NEUTRAL',
        r'PREDICTION[:\s]*NEUTRAL',
        r'SIDEWAYS',
        r'CONSOLIDAT',
        r'RANGE.?BOUND',
        r'NO\s+CLEAR\s+DIRECTION',
        r'MIXED\s+SIGNALS?',
    ]
    
    # Check patterns
    for pattern in bullish_patterns:
        if re.search(pattern, text_upper):
            return "BULLISH"
    
    for pattern in bearish_patterns:
        if re.search(pattern, text_upper):
            return "BEARISH"
    
    for pattern in neutral_patterns:
        if re.search(pattern, text_upper):
            return "NEUTRAL"
    
    # Count occurrences as fallback
    bullish_count = len(re.findall(r'\bBULLISH\b', text_upper))
    bearish_count = len(re.findall(r'\bBEARISH\b', text_upper))
    
    if bullish_count > bearish_count:
        return "BULLISH"
    elif bearish_count > bullish_count:
        return "BEARISH"
    
    return "NEUTRAL"


def extract_confidence(text: str) -> str:
    """Extract confidence level from text."""
    text_upper = text.upper()
    
    high_patterns = [
        r'CONFIDENCE[:\s]*HIGH',
        r'HIGH\s+CONFIDENCE',
        r'STRONGLY\s+(?:BULLISH|BEARISH)',
        r'VERY\s+(?:LIKELY|CONFIDENT)',
        r'HIGH\s+PROBABILITY',
    ]
    
    low_patterns = [
        r'CONFIDENCE[:\s]*LOW',
        r'LOW\s+CONFIDENCE',
        r'UNCERTAIN',
        r'NOT\s+(?:VERY\s+)?CONFIDENT',
        r'LOW\s+PROBABILITY',
        r'WEAK\s+(?:SIGNAL|SETUP)',
    ]
    
    for pattern in high_patterns:
        if re.search(pattern, text_upper):
            return "HIGH"
    
    for pattern in low_patterns:
        if re.search(pattern, text_upper):
            return "LOW"
    
    return "MEDIUM"


def extract_reasoning(text: str) -> str:
    """Extract reasoning section from response."""
    # Look for reasoning section
    patterns = [
        r'REASONING[:\s]*(.+?)(?=\n\n|\Z)',
        r'BRIEF REASONING[:\s]*(.+?)(?=\n\n|\Z)',
        r'ANALYSIS[:\s]*(.+?)(?=\n\n|\Z)',
        r'BECAUSE[:\s]*(.+?)(?=\n\n|\Z)',
        r'(?:THE\s+)?REASON[:\s]*(.+?)(?=\n\n|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            # Clean up
            reasoning = re.sub(r'\s+', ' ', reasoning)
            if len(reasoning) > 20:  # Minimum meaningful length
                return reasoning[:500]  # Cap at 500 chars
    
    # Fallback: take last paragraph if it looks like reasoning
    paragraphs = text.strip().split('\n\n')
    if paragraphs:
        last = paragraphs[-1].strip()
        if len(last) > 50 and not any(x in last.upper() for x in ['PRICE:', 'TARGET:', '$']):
            return last[:500]
    
    return ""


def parse_llm_response(response: str) -> ParsedAnalysis:
    """
    Parse LLM response into structured analysis.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        ParsedAnalysis with extracted data
    """
    logger.debug(f"Parsing LLM response ({len(response)} chars)")
    
    # Extract components
    direction = extract_direction(response)
    confidence = extract_confidence(response)
    
    # Extract prices
    price_1h = extract_price(response, r'(?:1\s*hour?|1h|1-hour)') or \
               extract_price(response, r'(?:expected|target).*?(?:1\s*h)')
    
    price_4h = extract_price(response, r'(?:4\s*hours?|4h|4-hour)') or \
               extract_price(response, r'(?:expected|target).*?(?:4\s*h)')
    
    invalidation = extract_price(response, r'invalidation') or \
                   extract_price(response, r'stop.?loss') or \
                   extract_price(response, r'(?:if\s+price\s+)?breaks?\s+(?:below|above)')
    
    # Key levels
    support = extract_price(response, r'(?:critical\s+)?support') or \
              extract_price(response, r'key\s+support')
    
    resistance = extract_price(response, r'(?:critical\s+)?resistance') or \
                 extract_price(response, r'key\s+resistance')
    
    # Reasoning
    reasoning = extract_reasoning(response)
    
    result = ParsedAnalysis(
        direction=direction,
        confidence=confidence,
        price_1h=price_1h,
        price_4h=price_4h,
        invalidation_level=invalidation,
        critical_support=support,
        critical_resistance=resistance,
        reasoning=reasoning,
        raw_response=response,
    )
    
    logger.debug(f"Parsed: {direction} ({confidence}), 1h=${price_1h}, 4h=${price_4h}")
    
    return result
