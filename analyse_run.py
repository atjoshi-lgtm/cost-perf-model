# I have a file in exp1/run_1 (actually take this as an argument) that contains the output of a run of the model. 
# There are lines like: Updated Objective Value: 13534.74 USD. Just extract these lines and plot the objective value over time. You can use matplotlib for plotting. Save the plot as a PNG file.
# Next, there are lines like Metro: LAX, P50: 24.827 ms, P95: 100.423 ms. Extract these lines and plot the P50 and P95 values for each metro over time. Plot P50 and P95 in seperate files. You can use different colors or markers for different metros. Save the plot as a PNG file.
# Next, there are lines like Disk provisioned for metro PHX: 79 TB. Extract these lines and plot the provisioned disk for each metro over time. Again, use different colors or markers for different metros. Save the plot as a PNG file.
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_objective_values(file_path):
    objective_values = []
    total_costs = []
    total_penalties = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(
                r'Updated Objective Value: ([\d.]+) USD, Total cost: ([\d.]+) USD, Total performance penalty: ([\d.]+) USD',
                line
            )
            if match:
                objective_values.append(float(match.group(1)))
                total_costs.append(float(match.group(2)))
                total_penalties.append(float(match.group(3)))
    return objective_values, total_costs, total_penalties

def extract_performance_metrics(file_path):
    performance_metrics = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Metro: (\w+), P50: ([\d.]+) ms, P95: ([\d.]+) ms', line)
            if match:
                metro = match.group(1)
                p50 = float(match.group(2))
                p95 = float(match.group(3))
                performance_metrics[metro].append((p50, p95))
    return performance_metrics

def extract_disk_provisioning(file_path):
    disk_provisioning = defaultdict(list)
    hitrates = defaultdict(list)
    costs = defaultdict(list)
    penalties = defaultdict(list)

    pattern = re.compile(
        r"Disk provisioned for metro (\w+):\s+(\d+)\s+TB,\s+Hitrate:\s+([\d.]+)%?,\s+Cost:\s+([\d.]+)\s+USD,\s+Performance Penalty:\s+([\d.]+)\s+USD"
    )

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                metro = match.group(1)
                disk_tb = float(match.group(2))
                hitrate = float(match.group(3))
                cost = float(match.group(4))
                penalty = float(match.group(5))

                disk_provisioning[metro].append(disk_tb)
                hitrates[metro].append(hitrate)
                costs[metro].append(cost)
                penalties[metro].append(penalty)

    return disk_provisioning, hitrates, costs, penalties

def plot_objective_values(objective_values, total_costs, total_penalties, output_file):
    plt.figure()
    plt.plot(objective_values, marker='o', label='Objective')
    plt.plot(total_costs, marker='s', label='Total cost')
    plt.plot(total_penalties, marker='^', label='Total performance penalty')
    plt.title('Objective, Cost, and Penalty Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('USD')
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.clf()

# Plot P50 and P95 in separate png files
def plot_performance_metrics(performance_metrics, output_file_p50, output_file_p95):
    plt.figure()
    for metro, metrics in performance_metrics.items():
        p50_values = [metric[0] for metric in metrics]
        plt.plot(p50_values, marker='o', label=metro)
    plt.title('P50 Values Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('P50 (ms)')
    plt.legend()
    plt.grid()
    plt.savefig(output_file_p50)
    plt.clf()

    plt.figure()
    for metro, metrics in performance_metrics.items():
        p95_values = [metric[1] for metric in metrics]
        plt.plot(p95_values, marker='o', label=metro)
    plt.title('P95 Values Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('P95 (ms)')
    plt.legend()
    plt.grid()
    plt.savefig(output_file_p95)
    plt.clf()

def plot_disk_provisioning(disk_provisioning, output_file):
    plt.figure()
    for metro, disks in disk_provisioning.items():
        plt.plot(disks, marker='o', label=metro)
    plt.title('Disk Provisioning Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Disk Provisioned (TB)')
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.clf()

def plot_disk_provisioning(disk_provisioning, output_file):
    plt.figure()
    for metro, values in disk_provisioning.items():
        plt.plot(values, marker="o", label=metro)
    plt.title("Disk Provisioned Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Disk (TB)")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.clf()


def plot_hitrates(hitrates, output_file):
    plt.figure()
    for metro, values in hitrates.items():
        plt.plot(values, marker="o", label=metro)
    plt.title("Hitrate Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Hitrate (%)")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.clf()


def plot_costs(costs, output_file):
    plt.figure()
    for metro, values in costs.items():
        plt.plot(values, marker="o", label=metro)
    plt.title("Per-metro Cost Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (USD)")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.clf()

def plot_penalties(penalties, output_file):
    plt.figure()
    for metro, values in penalties.items():
        plt.plot(values, marker="o", label=metro)
    plt.title("Per-metro Performance Penalty Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Penalty (USD)")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.clf()

def extract_hitrates(file_path):
    hitrates = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(
                r'Disk provisioned for metro (\w+): \d+ TB, Hitrate: ([\d.]+)',
                line
            )
            if match:
                metro = match.group(1)
                hitrate = float(match.group(2))
                hitrates[metro].append(hitrate)
    return hitrates

def plot_hitrates(hitrates, output_file):
    plt.figure()
    for metro, values in hitrates.items():
        plt.plot(values, marker='o', label=metro)
    plt.title('Hitrate Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Hitrate (%)')
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.clf()

if __name__ == "__main__":
    file_path = 'exp4/run_5'  # Change this to your actual file path

    # Extract run directory
    run_dir = file_path.split('/')[0]

    objective_values, total_costs, total_penalties = extract_objective_values(file_path)
    performance_metrics = extract_performance_metrics(file_path)
    disk_provisioning, hitrates, costs, penalties = extract_disk_provisioning(file_path)

    plot_objective_values(objective_values, total_costs, total_penalties, f'{run_dir}/objective_values.png')
    plot_performance_metrics(performance_metrics, f'{run_dir}/p50_values.png', f'{run_dir}/p95_values.png')
    plot_disk_provisioning(disk_provisioning, f'{run_dir}/disk_provisioning.png')
    plot_hitrates(hitrates, f'{run_dir}/hitrates.png')
    plot_costs(costs, f'{run_dir}/costs.png')
    plot_penalties(penalties, f'{run_dir}/penalties.png')