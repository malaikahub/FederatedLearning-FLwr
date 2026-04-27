import flwr as fl
import time

round_accuracies = []
start_time = time.time()


def weighted_average(metrics):
    accuracies = [m["accuracy"] for _, m in metrics]
    avg_accuracy = sum(accuracies) / len(accuracies)

    round_accuracies.append(avg_accuracy)

    print(f"Round {len(round_accuracies)} Accuracy: {avg_accuracy:.4f}")

    return {"accuracy": avg_accuracy}


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

end_time = time.time()

print("\n========= FINAL RESULTS =========")
print(f"Final Accuracy: {round_accuracies[-1]:.4f}")
print(f"Training Time: {end_time - start_time:.2f} seconds")