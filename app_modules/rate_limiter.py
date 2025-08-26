"""
Alpha Vantage API Rate Limiter for Stock Forecasting Tool

This module provides rate limiting functionality for Alpha Vantage API calls
to ensure compliance with API usage limits. It tracks API call timestamps
and enforces waiting periods when necessary.

Adapted from algo trading project utils/av_api_rate_limiter.py

Classes:
    AVAPIRateLimiter: Class-based rate limiter for Alpha Vantage API

Functions:
    rate_limit(): Simple function-based rate limiting

Author: Shane
Created: 2025-08-26
"""

import time
import logging
from collections import deque
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AVAPIRateLimiter:
    """
    Class-based Alpha Vantage API rate limiter.

    Supports different Alpha Vantage tiers:
    - Free: 5 calls per minute, 500 per day
    - Premium: 75 calls per minute, 150,000 per day
    """

    def __init__(self, max_calls_per_minute: int = 5):
        """
        Initialize the rate limiter.

        Args:
            max_calls_per_minute (int): Maximum API calls per minute.
                                      Default is 5 for AV Free tier.
                                      Use 75 for AV Premium.
        """
        self.call_timestamps = deque(maxlen=max_calls_per_minute)
        self.max_calls = max_calls_per_minute
        self.tier = "Premium" if max_calls_per_minute > 5 else "Free"

        logger.info(
            f"Alpha Vantage rate limiter initialized: {max_calls_per_minute} calls/minute ({self.tier} tier)"
        )

    def log_configuration(self):
        """Log the current rate limiter configuration."""
        logger.info(
            f"Rate limiting: {self.max_calls} calls per minute (Alpha Vantage {self.tier})"
        )
        status = self.get_status()
        logger.info(
            f"Current API calls in window: {status['current_calls_in_window']}/{status['max_calls_per_minute']}"
        )
        logger.info(f"Remaining capacity: {status['remaining_capacity']} calls")

    def wait_if_needed(self, min_sleep_time: float = 0.1) -> float:
        """
        Ensures that API calls adhere to the rate limit.
        This method blocks execution if the rate limit is about to be exceeded.

        Args:
            min_sleep_time (float): Minimum time to sleep in seconds if a wait is required.

        Returns:
            float: Actual sleep time in seconds (0 if no sleep was needed)
        """
        current_time = time.time()
        sleep_time = 0

        # If the deque is full, check the timestamp of the oldest call
        if len(self.call_timestamps) == self.call_timestamps.maxlen:
            oldest_call_timestamp = self.call_timestamps[0]
            # Calculate time elapsed since the oldest call in the deque
            elapsed_time_since_oldest_relevant_call = (
                current_time - oldest_call_timestamp
            )

            # If less than 60 seconds have passed since the oldest call, we need to wait
            if elapsed_time_since_oldest_relevant_call < 60:
                # Calculate the remaining time to wait to hit the 60-second mark
                wait_duration = 60 - elapsed_time_since_oldest_relevant_call
                # Sleep for the calculated duration, ensuring a minimum sleep time
                sleep_time = max(
                    min_sleep_time, wait_duration + 0.1
                )  # Add a small buffer

                logger.info(
                    f"Rate limit approaching ({self.max_calls} calls/min). "
                    f"Sleeping for {sleep_time:.2f} seconds to comply."
                )
                time.sleep(sleep_time)
            else:
                logger.debug("No rate limit sleep needed (past 60s window).")
        else:
            logger.debug("Deque not full, no immediate rate limit check required.")

        # Record the current API call timestamp
        self.call_timestamps.append(time.time())
        logger.debug(
            f"API call timestamp recorded. Deque size: {len(self.call_timestamps)}"
        )

        return sleep_time

    def get_status(self) -> Dict[str, Any]:
        """
        Get current rate limiter status.

        Returns:
            dict: Status information including current call count and remaining capacity.
        """
        current_calls = len(self.call_timestamps)
        return {
            "current_calls_in_window": current_calls,
            "max_calls_per_minute": self.max_calls,
            "remaining_capacity": self.max_calls - current_calls,
            "tier": self.tier,
        }

    def estimate_time_for_calls(self, num_calls: int) -> float:
        """
        Estimate the time required to make a given number of API calls
        considering rate limiting.

        Args:
            num_calls (int): Number of API calls to estimate for

        Returns:
            float: Estimated time in minutes
        """
        if num_calls <= 0:
            return 0

        # Calculate batches needed
        batches = (num_calls + self.max_calls - 1) // self.max_calls

        # Each batch takes about 1 minute (60 seconds) due to rate limiting
        # Plus some buffer time
        estimated_minutes = batches * 1.1  # 10% buffer

        return estimated_minutes


# Global rate limiter instance (can be initialized once and reused)
_global_rate_limiter = None


def get_rate_limiter(max_calls_per_minute: int = 5) -> AVAPIRateLimiter:
    """
    Get or create the global rate limiter instance.

    Args:
        max_calls_per_minute (int): Maximum calls per minute

    Returns:
        AVAPIRateLimiter: Rate limiter instance
    """
    global _global_rate_limiter

    if (
        _global_rate_limiter is None
        or _global_rate_limiter.max_calls != max_calls_per_minute
    ):
        _global_rate_limiter = AVAPIRateLimiter(max_calls_per_minute)

    return _global_rate_limiter


def rate_limit(max_calls_per_minute: int = 5, min_sleep_time: float = 0.1) -> float:
    """
    Simple function-based rate limiting for Alpha Vantage API calls.

    Args:
        max_calls_per_minute (int): Maximum API calls per minute
        min_sleep_time (float): Minimum sleep time in seconds

    Returns:
        float: Actual sleep time in seconds
    """
    limiter = get_rate_limiter(max_calls_per_minute)
    return limiter.wait_if_needed(min_sleep_time)


if __name__ == "__main__":
    # Test the rate limiter
    logging.basicConfig(level=logging.INFO)

    print("Testing Alpha Vantage Rate Limiter...")

    # Test with free tier (5 calls per minute)
    limiter = AVAPIRateLimiter(max_calls_per_minute=5)
    limiter.log_configuration()

    print(
        f"Estimating time for 20 calls: {limiter.estimate_time_for_calls(20):.1f} minutes"
    )

    # Simulate some API calls
    for i in range(8):
        print(f"Making simulated call {i+1}")
        sleep_time = limiter.wait_if_needed()
        if sleep_time > 0:
            print(f"  Slept for {sleep_time:.2f} seconds")

        status = limiter.get_status()
        print(
            f"  Status: {status['current_calls_in_window']}/{status['max_calls_per_minute']} calls used"
        )

        # Simulate some processing time
        time.sleep(0.1)

    print("Rate limiter test complete!")
