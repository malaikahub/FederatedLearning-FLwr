# Federated Learning Simulation using Flower (FLwr)

> Privacy-preserving distributed machine learning on MNIST — comparing **FedAvg**, **FedProx**, and **Centralized** training.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [What is Federated Learning?](#what-is-federated-learning)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Key Observations](#key-observations)
- [Centralized vs Federated](#centralized-vs-federated)
- [Conclusion](#conclusion)
- [Authors](#authors)

---

##  Project Overview

This project demonstrates **Federated Learning** using the [Flower (FLwr)](https://flower.dev/) framework. Multiple simulated clients train a neural network on the **MNIST handwritten digit dataset** — without ever sharing their raw data with the server.

Three training strategies are implemented and compared:

| Strategy | Type | Description |
|----------|------|-------------|
| **FedAvg** | Federated | Standard federated averaging |
| **FedProx** | Federated | FedAvg + proximal regularization |
| **Centralized** | Baseline | All data in one place, no federation |

---

##  What is Federated Learning?

Federated Learning is a distributed machine learning approach where training happens **on-device** rather than on a central server.

```
Each Client:                    Server:
┌─────────────────┐            ┌─────────────────────┐
│  Local Data     │            │  Aggregate Updates   │
│  Local Training │ ─updates─▶ │  (FedAvg / FedProx) │
│  No data leaves │ ◀─model──  │  Global Model        │
└─────────────────┘            └─────────────────────┘
```

**Key principles:**
-  Raw data **never leaves** the client device
-  Only model weights/gradients are transmitted
-  Server aggregates updates into a single global model
-  Process repeats for multiple rounds

**Benefits:**
-  Strong data privacy
-  Reduced data transfer overhead
-  Works with distributed/heterogeneous data
-  Compliant with data regulations (GDPR etc.)

---

##  Technologies Used

| Tool | Purpose |
|------|---------|
| `Python 3.x` | Core programming language |
| `Flower (FLwr)` | Federated learning framework |
| `TensorFlow / Keras` | Neural network training |
| `NumPy` | Numerical computations |
| `Matplotlib` | Result visualization |
| `MNIST Dataset` | Handwritten digit benchmark |
| `VS Code` | Development environment |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FLOWER SERVER                     │
│           (FedAvg / FedProx Aggregation)            │
└──────────┬──────────────────────────────┬───────────┘
           │  broadcast global model      │ receive updates
           ▼                              │
┌──────────────────────────────────────────────────────┐
│              CLIENTS  (Simulated)                    │
│                                                      │
│  Client 0   Client 1   Client 2   Client 3  Client 4 │
│  [MNIST]    [MNIST]    [MNIST]    [MNIST]   [MNIST]  │
│  shard 0    shard 1    shard 2    shard 3   shard 4  │
│                                                      │
│  Each trains locally → sends only model updates ────▶│
└──────────────────────────────────────────────────────┘
```

**Training Flow:**
1. Server initializes global model
2. Server sends model to all clients
3. Each client trains on its local shard
4. Clients send model updates back to server
5. Server aggregates updates (FedAvg / FedProx)
6. Repeat for N rounds

---

##  Project Structure

```
FederatedLearning_FLWR/
│
├──  server.py            # Flower server — FedAvg strategy
├──  fedprox.py           # FedProx strategy implementation
├──  client.py            # Flower client — local training
├──  centralized.py       # Baseline centralized training
├──  model.py             # Neural network architecture
├──  dataset.py           # MNIST loading & partitioning
├──  plots.py             # Result visualization scripts
│
├── plot1_accuracy.png           # Round-wise accuracy
├── plot2_loss.png               # Round-wise loss
├── plot3_final_accuracy_bar.png # Final accuracy comparison
├── plot4_training_time.png      # Training time comparison
├── plot5_dashboard.png          # Full results dashboard
│
├──  server_log.txt       # Server training logs
└──  client_logs/         # Per-client training logs
```

---

##  How to Run

### 1. Install Dependencies

```bash
pip install flwr tensorflow numpy matplotlib
```

### 2. Run FedAvg (Federated Averaging)

Open **6 terminals** and run in order:

**Terminal 1 — Start the server:**
```bash
python server.py
```

**Terminals 2–6 — Start 5 clients (one per terminal):**
```bash
python client.py 0
python client.py 1
python client.py 2
python client.py 3
python client.py 4
```

### 3. Run FedProx

```bash
python fedprox.py
```

### 4. Run Centralized Training (Baseline)

```bash
python centralized.py
```

### 5. Generate All Plots

```bash
python plots.py
```

>  **Tip:** Make sure the server is running before starting any clients. Clients will automatically connect to `localhost:8080`.

---

## 📊 Results

### Final Performance Summary

| Method | Final Accuracy | Final Loss | Training Time | Privacy |
|--------|---------------|------------|---------------|---------|
| **FedProx** | **99.95%** | **0.0060** | 174.39 sec |  Yes |
| **FedAvg** | 99.88% | 0.0078 | 148.66 sec |  Yes |
| **Centralized** | 99.57% | — | 82.44 sec |  No |

### Round-wise Accuracy

| Round | FedAvg | FedProx | Centralized |
|-------|--------|---------|-------------|
| 1 | 96.56% | 96.67% | 95.22% |
| 2 | 98.64% | 98.77% | 98.34% |
| 3 | 99.48% | 99.45% | 98.90% |
| 4 | 99.77% | 99.82% | 99.30% |
| 5 | **99.88%** | **99.95%** | **99.57%** |

### Round-wise Loss (Federated)

| Round | FedAvg | FedProx |
|-------|--------|---------|
| 1 | 0.1409 | 0.1356 |
| 2 | 0.0551 | 0.0518 |
| 3 | 0.0266 | 0.0252 |
| 4 | 0.0139 | 0.0123 |
| 5 | **0.0078** | **0.0060** |

---

##  Key Observations

**Accuracy**
- FedProx achieved the **highest accuracy at 99.95%** — beating even centralized training
- FedAvg closely followed at 99.88%
- Centralized training achieved 99.57% despite having access to all data

**Speed**
- Centralized is fastest (82.44 sec) — no communication overhead
- FedAvg takes moderate time (148.66 sec) — aggregation rounds add cost
- FedProx is slowest (174.39 sec) — proximal regularization adds computation

**Convergence**
- FedProx converges more smoothly due to the proximal term keeping clients close to the global model
- Both federated methods showed consistent improvement every round

---

##  Centralized vs Federated Learning

| Feature | Centralized | Federated Learning |
|---------|-------------|-------------------|
| **Data location** | Single server | Stays on each client |
| **Privacy** |  Low |  High |
| **Training speed** |  Fast |  Slower |
| **Final accuracy** | Good (99.57%) | Better (up to 99.95%) |
| **Communication** | None | Required each round |
| **Scalability** | Limited | Highly scalable |
| **Data regulation** | Harder to comply | GDPR-friendly |

---

##  Conclusion

This project successfully demonstrates that **Federated Learning can match or exceed centralized training accuracy** while keeping data fully private and distributed.

-  **FedProx** → Best accuracy (99.95%) and lowest loss — recommended when accuracy is the priority
-  **FedAvg** → Great balance between speed, accuracy, and simplicity — best general-purpose choice
-  **Centralized** → Fastest to train but offers no privacy guarantees

> Federated Learning is the future of privacy-preserving AI — especially for healthcare, finance, and mobile applications where data cannot leave the device.

---

##  Authors

| Name | Roll No |
|------|---------|
| Tehreem Ejaz | 089 |
| Zainab Saleem | 093 |
| Malaika Arooj| 057 |
| Maryam | 061 |

**Presentation Date:** 2 June, 2026  
**Framework:** [Flower (FLwr)](https://flower.dev/) | **Dataset:** MNIST \




</div>
