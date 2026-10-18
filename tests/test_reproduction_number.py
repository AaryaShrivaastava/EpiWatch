"""Tests for Reproduction Number."""
import pytest
from src.epidemiology.reproduction_number import calculate_r0_sir, calculate_r0_sird, estimate_rt_naive

def test_calculate_r0_sir():
    assert calculate_r0_sir(0.3, 0.1) == pytest.approx(3.0)
    assert calculate_r0_sir(0.0, 0.1) == pytest.approx(0.0)
    assert calculate_r0_sir(0.3, 0.0) == float('inf')

def test_calculate_r0_sird():
    assert calculate_r0_sird(0.3, 0.1, 0.05) == pytest.approx(0.3 / 0.15)
    assert calculate_r0_sird(0.3, 0.0, 0.0) == float('inf')

def test_estimate_rt_naive():
    assert estimate_rt_naive(0.3, 500, 1000, 0.1) == pytest.approx(1.5)
