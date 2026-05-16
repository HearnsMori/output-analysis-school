# M/M/1 Queue Simulation & Output Analysis

This project simulates an **M/M/1 queueing system** (Arrival rate $\lambda = 0.8$, Service rate $\mu = 1.0$) and applies three foundational stochastic output analysis methods to estimate the true steady-state mean waiting time. 

Because simulations often begin in an unrepresentative state (e.g., an empty queue), these methods are critical for removing initialization bias ("warm-up" period) and accounting for autocorrelation.

## 📊 Analyzed Methods

1. **Welch Method**: Averages across multiple initial replications, applies a moving average window, and visualizes where the transient phase transitions into steady-state stability.
2. **Replication-Deletion**: Deletes a fixed percentage (20%) of the initial warm-up data from independent replications to establish an unbiased grand mean.
3. **Batch Means**: Takes a single, long continuous simulation run, strips the warm-up period, and cuts the remaining observations into contiguous, independent batches to compute a robust confidence interval.

---

## 🚀 Setup & Installation

### 1. Clone
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
### 2. Run
```bash
python output_analysis.py
```