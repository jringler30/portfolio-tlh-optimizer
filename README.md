# Streamlit Trade TLH Optimizer

This project is a Streamlit-based portfolio simulation and tax-aware optimization tool.

It supports two execution modes:

## Standard Portfolio Engine
- Buy-and-hold simulation
- Scheduled rebalancing
- Price-based portfolio valuation

## Optimizer MSBA v1
- Trade-driven accounting engine
- Tax lot tracking
- Short-term vs long-term gain classification
- Tax-loss harvesting (threshold-based)
- Dividend processing (cash or reinvestment)
- Static vs optimized portfolio comparison

---

## Data Requirements

### Price dataset must include:

- TRADINGITEMID  
- TICKERSYMBOL  
- PRICEDATE  
- PRICECLOSE  

### Dividend dataset must include:

- TRADINGITEMID  
- PAYDATE  
- DIVAMOUNT  

---

## Run locally

pip install -r requirements.txt
streamlit run portfolio_returns_engine_code.py

---

## Known Limitations (MSBA v1)

- Wash-sale rules are not enforced
- Dividend taxes not separately modeled
- Transaction costs/slippage not included
