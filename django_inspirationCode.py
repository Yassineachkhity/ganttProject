import pulp as lp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

class JobShopProblem:
    def __init__(self, n_jobs: int, n_machines: int):
        """
        Initialise un problème de Job Shop
        Args:
            n_jobs: Nombre de jobs
            n_machines: Nombre de machines
        """
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.operations = []  # Liste des opérations par job
        self.processing_times = []  # Temps de traitement
        self.machine_sequence = []  # Séquence des machines

    def add_job(self, operations: List[Tuple[int, float]]):
        """
        Ajoute un job avec ses opérations
        Args:
            operations: Liste de tuples (machine_id, processing_time)
        """
        self.operations.append(operations)
        self.processing_times.append([op[1] for op in operations])
        self.machine_sequence.append([op[0] for op in operations])

def solve_milp_jobshop(problem: JobShopProblem) -> Dict:
    """
    Résout le problème de Job Shop en utilisant MILP
    Args:
        problem: Instance de JobShopProblem
    Returns:
        Dict contenant la solution
    """
    # Création du modèle
    model = lp.LpProblem("JobShop", lp.LpMinimize)
    
    # Paramètres
    n_jobs = problem.n_jobs
    n_machines = problem.n_machines
    processing_times = problem.processing_times
    machine_sequence = problem.machine_sequence
    
    # Variables
    # Temps de début de chaque opération
    start_times = lp.LpVariable.dicts("start",
                                    ((i, j) for i in range(n_jobs) 
                                     for j in range(len(problem.operations[i]))),
                                    lowBound=0)
    
    # Variable binaire pour l'ordre des opérations sur les machines
    y = lp.LpVariable.dicts("y",
                           ((i1, j1, i2, j2) for i1 in range(n_jobs)
                            for j1 in range(len(problem.operations[i1]))
                            for i2 in range(n_jobs)
                            for j2 in range(len(problem.operations[i2]))
                            if i1 != i2),
                           cat='Binary')
    
    # Variable pour le makespan
    makespan = lp.LpVariable("Makespan", lowBound=0)
    
    # Objectif : minimiser le makespan
    model += makespan
    
    # Contraintes
    # 1. Contraintes de précédence entre opérations d'un même job
    for i in range(n_jobs):
        for j in range(len(problem.operations[i])-1):
            model += start_times[i,j+1] >= start_times[i,j] + processing_times[i][j]
    
    # 2. Contraintes de non-chevauchement sur les machines
    M = sum([sum(pt) for pt in processing_times]) # Big-M
    for i1 in range(n_jobs):
        for j1 in range(len(problem.operations[i1])):
            for i2 in range(i1+1, n_jobs):
                for j2 in range(len(problem.operations[i2])):
                    if machine_sequence[i1][j1] == machine_sequence[i2][j2]:
                        model += start_times[i1,j1] + processing_times[i1][j1] <= \
                                start_times[i2,j2] + M * (1-y[i1,j1,i2,j2])
                        model += start_times[i2,j2] + processing_times[i2][j2] <= \
                                start_times[i1,j1] + M * y[i1,j1,i2,j2]
    
    # 3. Contrainte de makespan
    for i in range(n_jobs):
        model += makespan >= start_times[i,len(problem.operations[i])-1] + \
                           processing_times[i][-1]
    
    # Résolution
    model.solve()
    
    # Extraction de la solution
    solution = {
        "status": lp.LpStatus[model.status],
        "makespan": lp.value(makespan),
        "schedule": {}
    }
    
    for i in range(n_jobs):
        solution["schedule"][i] = []
        for j in range(len(problem.operations[i])):
            solution["schedule"][i].append({
                "start_time": lp.value(start_times[i,j]),
                "machine": machine_sequence[i][j],
                "processing_time": processing_times[i][j]
            })
    
    return solution

def create_gantt_chart(solution: Dict, problem: JobShopProblem, save_path: str = None):
    """
    Crée un diagramme de Gantt pour la solution
    Args:
        solution: Solution du problème
        problem: Instance du problème
        save_path: Chemin pour sauvegarder l'image
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Couleurs pour chaque job
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, problem.n_jobs))
    
    # Création du Gantt
    for job_id, operations in solution["schedule"].items():
        for op in operations:
            ax.barh(y=op["machine"],
                   width=op["processing_time"],
                   left=op["start_time"],
                   color=colors[job_id],
                   alpha=0.8,
                   label=f'Job {job_id}' if op == operations[0] else "")
    
    # Personnalisation du graphique
    ax.set_xlabel('Temps')
    ax.set_ylabel('Machine')
    ax.set_title('Diagramme de Gantt - Job Shop')
    ax.grid(True)
    
    # Légende
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def jackson_rule_2machines(processing_times_m1: List[float], 
                         processing_times_m2: List[float]) -> List[int]:
    """
    Implémente la règle de Jackson pour 2 machines
    Args:
        processing_times_m1: Temps de traitement sur la machine 1
        processing_times_m2: Temps de traitement sur la machine 2
    Returns:
        Séquence optimale des jobs
    """
    n = len(processing_times_m1)
    jobs = list(range(n))
    
    # Calcul des ratios pour chaque job
    ratios = [(i, processing_times_m1[i]/processing_times_m2[i]) 
              for i in jobs]
    
    # Tri selon la règle de Johnson
    sorted_jobs = sorted(ratios, key=lambda x: x[1])
    return [job[0] for job in sorted_jobs]

def calculate_makespan(schedule: Dict, problem: JobShopProblem) -> float:
    """
    Calcule le makespan d'une solution
    Args:
        schedule: Planning des opérations
        problem: Instance du problème
    Returns:
        Makespan de la solution
    """
    max_completion_time = 0
    for job_id, operations in schedule.items():
        completion_time = operations[-1]["start_time"] + operations[-1]["processing_time"]
        max_completion_time = max(max_completion_time, completion_time)
    return max_completion_time

def evaluate_solution(solution: Dict, problem: JobShopProblem) -> Dict:
    """
    Évalue une solution selon plusieurs critères
    Args:
        solution: Solution à évaluer
        problem: Instance du problème
    Returns:
        Dict avec les métriques d'évaluation
    """
    metrics = {
        "makespan": solution["makespan"],
        "machine_utilization": {},
        "job_waiting_times": {}
    }
    
    # Calcul de l'utilisation des machines
    total_time = solution["makespan"]
    machine_busy_time = {m: 0 for m in range(problem.n_machines)}
    
    for job_id, operations in solution["schedule"].items():
        for op in operations:
            machine_busy_time[op["machine"]] += op["processing_time"]
    
    for machine, busy_time in machine_busy_time.items():
        metrics["machine_utilization"][machine] = busy_time / total_time
    
    # Calcul des temps d'attente des jobs
    for job_id, operations in solution["schedule"].items():
        waiting_time = 0
        last_completion = 0
        for op in operations:
            waiting_time += max(0, op["start_time"] - last_completion)
            last_completion = op["start_time"] + op["processing_time"]
        metrics["job_waiting_times"][job_id] = waiting_time
    
    return metrics
