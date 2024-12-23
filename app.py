from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, FormField, FieldList, SelectField
from wtforms.validators import DataRequired, NumberRange
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import os
import random
from pulp import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['DEBUG'] = True  # Enable debug mode to get more detailed errors
app.config['WTF_CSRF_ENABLED'] = True

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/jobshop')
def jobshop():
    return render_template('jobshop.html')

# @app.route('/jobshop2')
# def jobshop2():
#     return render_template('jobshop2.html')

# @app.route('/jobshop3')
# def jobshop3():
#     return render_template('jobshop3.html')


@app.route('/flowshop')
def flowshop():
    return render_template('flowshop.html')

def apply_cds_rule(processing_times):
    """Implements Campbell, Dudek & Smith (CDS) rule for m machines."""
    num_jobs, num_machines = processing_times.shape
    best_makespan = float('inf')
    best_sequence = None
    
    # For k stages
    for k in range(1, num_machines):
        # Calculate aggregated processing times
        p1 = np.sum(processing_times[:, :k], axis=1)
        p2 = np.sum(processing_times[:, -k:], axis=1)
        
        # Apply Johnson's rule on the aggregated times
        jobs = list(range(num_jobs))
        set1 = [(i, t) for i, t in enumerate(p1) if t <= p2[i]]
        set2 = [(i, t) for i, t in enumerate(p1) if t > p2[i]]
        
        set1.sort(key=lambda x: x[1])  # Sort set1 by p1 times
        set2.sort(key=lambda x: -p2[x[0]])  # Sort set2 by p2 times in descending order
        
        sequence = [job[0] for job in set1 + set2]
        
        # Calculate makespan for this sequence
        ordered_times = processing_times[sequence]
        completion_times = np.zeros_like(ordered_times)
        
        # Calculate completion times
        for i in range(num_jobs):
            for j in range(num_machines):
                if i == 0 and j == 0:
                    completion_times[i][j] = ordered_times[i][j]
                elif i == 0:
                    completion_times[i][j] = completion_times[i][j-1] + ordered_times[i][j]
                elif j == 0:
                    completion_times[i][j] = completion_times[i-1][j] + ordered_times[i][j]
                else:
                    completion_times[i][j] = max(completion_times[i][j-1], completion_times[i-1][j]) + ordered_times[i][j]
        
        makespan = float(completion_times[-1][-1])
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = sequence
    
    # Convert sequence to 1-based indexing and ensure Python int type
    return [int(i + 1) for i in (best_sequence if best_sequence is not None else range(num_jobs))]


def apply_johnson_rule(processing_times):
    """
    Applique la règle de Johnson pour deux machines.
    Args:
        processing_times: Temps de traitement pour chaque job sur chaque machine
    Returns:
        sequence: Liste des jobs ordonnés selon la règle de Johnson (1-indexed)
    """
    M1 = processing_times[0]
    M2 = processing_times[1]
    tasks = [(x, y) for x, y in zip(M1, M2)]
    tasks = list(enumerate(tasks, start=1))  # Ajoute les indices des tâches
    
    # Séparer les tâches en deux ensembles selon la règle de Johnson
    U = [task for task in tasks if task[1][0] < task[1][1]]
    V = [task for task in tasks if task[1][0] >= task[1][1]]
    
    # Trier U selon les temps de M1 (croissant) et V selon les temps de M2 (décroissant)
    U.sort(key=lambda x: x[1][0])
    V.sort(key=lambda x: x[1][1], reverse=True)
    
    # Combiner les deux ensembles pour obtenir la séquence optimale
    sigma = U + V
    return [task[0] for task in sigma]

def apply_cds_rule(processing_times):
    """
    Applique la règle CDS (Campbell, Dudek & Smith) pour m machines.
    Si le nombre de machines est 2, applique directement la règle de Johnson.
    Args:
        processing_times: Temps de traitement pour chaque job sur chaque machine
    Returns:
        best_sequence: Meilleure séquence trouvée (1-indexed)
    """
    processing_times = np.array(processing_times)
    num_machines = len(processing_times)
    num_jobs = len(processing_times[0])
    
    # Si on a seulement 2 machines, appliquer directement la règle de Johnson
    if num_machines == 2:
        return apply_johnson_rule(processing_times)
    
    best_makespan = float('inf')
    best_sequence = None
    
    # Pour chaque k de 1 à m-1, créer deux machines virtuelles
    for k in range(1, num_machines):
        # Créer les machines virtuelles G1 et G2
        G1 = [sum(processing_times[i][j] for i in range(k)) for j in range(num_jobs)]
        G2 = [sum(processing_times[i][j] for i in range(num_machines-k, num_machines)) for j in range(num_jobs)]
        
        # Appliquer la règle de Johnson sur les machines virtuelles
        virtual_times = np.array([G1, G2])
        sequence = apply_johnson_rule(virtual_times)
        
        # Calculer le makespan pour cette séquence
        ordered_times = processing_times[:, [i-1 for i in sequence]]
        completion_times = np.zeros((num_machines, num_jobs))
        
        # Calculer les temps de complétion
        for i in range(num_machines):
            for j in range(num_jobs):
                if i == 0 and j == 0:
                    completion_times[i][j] = ordered_times[i][j]
                elif i == 0:
                    completion_times[i][j] = completion_times[i][j-1] + ordered_times[i][j]
                elif j == 0:
                    completion_times[i][j] = completion_times[i-1][j] + ordered_times[i][j]
                else:
                    completion_times[i][j] = max(completion_times[i][j-1], completion_times[i-1][j]) + ordered_times[i][j]
        
        makespan = completion_times[-1][-1]
        
        # Mettre à jour la meilleure séquence si nécessaire
        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = sequence
    
    return best_sequence if best_sequence is not None else list(range(1, num_jobs + 1))


@app.route('/solve', methods=['GET', 'POST'])
def solve():
    if request.method == 'POST':
        try:
            # Retrieve data from the form
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data received"}), 400
                
            rule = data.get("rule")
            num_jobs = int(data.get("num_jobs"))
            num_machines = int(data.get("num_machines"))
            no_wait = data.get("no_wait", False)
            no_idle = data.get("no_idle", False)
            blocking = data.get("blocking", False)
            processing_times = data.get("processing_times")
            
            if not processing_times or len(processing_times) != num_jobs:
                return jsonify({"error": "Invalid processing times data"}), 400
                
            # Convert processing times to numpy array
            processing_times = np.array(processing_times)
            
            if processing_times.shape != (num_jobs, num_machines):
                return jsonify({"error": f"Processing times should be a {num_jobs}x{num_machines} matrix"}), 400
            
            # Perform scheduling based on the selected rule
            if rule == "spt":
                job_order = np.argsort(np.sum(processing_times, axis=1))  # Order by shortest processing time
            elif rule == "lpt":
                total_times = np.sum(processing_times, axis=1)
                job_order = np.argsort(-total_times)
            elif rule == "edd":
                # For EDD, we'll use total processing time as due date estimation
                due_dates = np.sum(processing_times, axis=1)
                job_order = np.argsort(due_dates)  # Order by earliest due date
            elif rule == "cds":
                job_order = [i - 1 for i in apply_cds_rule(processing_times)]  # Convert back to 0-based indexing
            elif rule == "johnson":
                if len(processing_times) != 2:
                    return jsonify({"error": "La règle de Johnson ne peut être appliquée que pour 2 machines"}), 400
                job_order = apply_johnson_rule(processing_times)
            elif rule == "fifo":
                job_order = np.arange(num_jobs)  # FIFO: order as they arrive
            elif rule == "lifo":
                job_order = np.arange(num_jobs)[::-1]  # LIFO: reverse order of arrival
            else:
                return jsonify({"error": "Invalid scheduling rule selected"}), 400

            # Calculate schedule
            start_times = np.zeros((num_jobs, num_machines))
            completion_times = np.zeros((num_jobs, num_machines))
            
            # Apply job order to processing times
            ordered_times = processing_times[job_order]
            
            # Calculate start and completion times
            for i in range(num_jobs):
                for j in range(num_machines):
                    if i == 0 and j == 0:
                        start_times[i][j] = 0
                    elif i == 0:
                        start_times[i][j] = completion_times[i][j-1]
                    elif j == 0:
                        start_times[i][j] = completion_times[i-1][j]
                    else:
                        start_times[i][j] = max(completion_times[i][j-1], completion_times[i-1][j])
                    
                    completion_times[i][j] = start_times[i][j] + ordered_times[i][j]
            
            # Apply constraints if needed
            if no_wait:
                start_times, completion_times = apply_no_wait_constraint(job_order, ordered_times)
            if no_idle:
                start_times, completion_times = apply_no_idle_constraint(job_order, ordered_times)
            if blocking:
                start_times, completion_times = apply_blocking_constraint(job_order, ordered_times)
            
            # Calculate performance metrics
            makespan = float(completion_times[-1][-1])
            total_flow_time = float(np.sum(completion_times[:, -1]))
            avg_flow_time = total_flow_time / num_jobs
            tfr = total_flow_time / (makespan * num_jobs)
            tar = avg_flow_time / makespan
            
            # Calculate machine utilization
            machine_utilization = calculate_machine_utilization(start_times, completion_times, makespan)
            idle_times = calculate_idle_times(start_times, completion_times, makespan)
            
            return jsonify({
                "sequence": [int(j + 1) for j in job_order],  # Convert to 1-based indexing
                "start_times": start_times.tolist(),
                "completion_times": completion_times.tolist(),
                "makespan": makespan,
                "total_flow_time": total_flow_time,
                "avg_flow_time": avg_flow_time,
                "tfr": tfr,
                "tar": tar,
                "machine_utilization": machine_utilization.tolist(),
                "idle_times": idle_times.tolist()
            })

        except Exception as e:
            print(f"Error in solve: {str(e)}")  # Add logging
            return jsonify({"error": str(e)}), 400

    return render_template('solve.html')


def calculate_performance_metrics(completion_times):
    """Calculate various performance metrics for the schedule."""
    num_jobs = len(completion_times)
    
    # Calculate Cmax (makespan)
    makespan = float(completion_times[-1][-1])
    
    # Calculate total flow time
    total_flow_time = sum(job_times[-1] for job_times in completion_times)
    
    # Calculate average flow time
    avg_flow_time = total_flow_time / num_jobs
    
    # Calculate TFR (Total Flow Ratio)
    tfr = total_flow_time / makespan
    
    # Calculate TAR (Time in Average Ratio)
    tar = avg_flow_time / makespan
    
    return {
        "makespan": makespan,
        "total_flow_time": total_flow_time,
        "avg_flow_time": avg_flow_time,
        "tfr": tfr,
        "tar": tar
    }


def generate_gantt_chart(job_order, processing_times, start_times):
    """Generates a Gantt chart and returns it as a base64 image."""
    fig, ax = plt.subplots(figsize=(12, 6))
    num_jobs, num_machines = processing_times.shape
    colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))

    # Calculate the maximum completion time for setting x-axis limit
    max_completion_time = np.max(start_times + processing_times)

    for i, job in enumerate(job_order):
        for m in range(num_machines):
            # Draw the job block
            bar = ax.barh(f'Machine {m + 1}', 
                         processing_times[i, m],
                         left=start_times[i, m],
                         color=colors[i],
                         edgecolor='black',
                         alpha=0.7)
            
            # Add job number to each bar if wide enough
            bar_width = processing_times[i, m]
            if bar_width > max_completion_time * 0.05:  # Only add text if bar is wide enough
                ax.text(start_times[i, m] + bar_width/2,
                       m + 0.5,
                       f'J{job+1}',
                       ha='center',
                       va='center',
                       color='white',
                       fontweight='bold',
                       fontsize=10)

    # Customize the chart
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0, max_completion_time * 1.05)  # Add 5% padding
    ax.invert_yaxis()
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, 
                                   label=f'Job {job+1}') for i, job in enumerate(job_order)]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()

    # Save the chart as a base64 string with high DPI
    img = BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    base64_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)
    return base64_img


def calculate_makespan(sequence, processing_times, setup_times=None):
    """Calculate makespan for a given sequence."""
    num_jobs = len(sequence)
    num_machines = processing_times.shape[1]
    completion_times = np.zeros((num_jobs, num_machines))
    
    # First job on first machine
    completion_times[0][0] = processing_times[sequence[0]][0]
    
    # First job on remaining machines
    for m in range(1, num_machines):
        completion_times[0][m] = completion_times[0][m-1] + processing_times[sequence[0]][m]
    
    # Remaining jobs
    for j in range(1, num_jobs):
        # First machine
        if setup_times is not None:
            setup_time = setup_times[sequence[j-1]][sequence[j]][0]
        else:
            setup_time = 0
        completion_times[j][0] = completion_times[j-1][0] + processing_times[sequence[j]][0] + setup_time
        
        # Remaining machines
        for m in range(1, num_machines):
            if setup_times is not None:
                setup_time = setup_times[sequence[j-1]][sequence[j]][m]
            else:
                setup_time = 0
            completion_times[j][m] = max(completion_times[j][m-1], completion_times[j-1][m]) + processing_times[sequence[j]][m] + setup_time
    
    return completion_times[-1][-1]  # Return makespan


def apply_spt_rule(processing_times):
    """Shortest Processing Time (SPT) rule."""
    total_times = np.sum(processing_times, axis=1)
    # Convert numpy array to Python list and add 1 for 1-based indexing
    return [int(i + 1) for i in np.argsort(total_times)]


def apply_lpt_rule(processing_times):
    """Longest Processing Time (LPT) rule."""
    total_times = np.sum(processing_times, axis=1)
    return [int(i + 1) for i in np.argsort(-total_times)]


def apply_edd_rule(due_dates):
    """Earliest Due Date (EDD) rule."""
    return [int(i + 1) for i in np.argsort(due_dates)]


def apply_fifo_rule(arrival_times):
    """First In First Out (FIFO) rule."""
    return [int(i + 1) for i in np.argsort(arrival_times)]


def apply_lifo_rule(arrival_times):
    """Last In First Out (LIFO) rule."""
    return [int(i + 1) for i in np.argsort(-arrival_times)]


def FIFO(release_times):
    """
    First In First Out (FIFO) rule for flow shop scheduling
    Args:
        release_times: list of release times for each job
    Returns:
        sequence: list of jobs ordered by FIFO rule (1-indexed)
    """
    jobs = list(range(1, len(release_times) + 1))
    # Sort jobs by release time (ascending)
    sorted_jobs = [x for _, x in sorted(zip(release_times, jobs))]
    return sorted_jobs

def LIFO(release_times):
    """
    Last In First Out (LIFO) rule for flow shop scheduling
    Args:
        release_times: list of release times for each job
    Returns:
        sequence: list of jobs ordered by LIFO rule (1-indexed)
    """
    jobs = list(range(1, len(release_times) + 1))
    # Sort jobs by release time (descending)
    sorted_jobs = [x for _, x in sorted(zip(release_times, jobs), reverse=True)]
    return sorted_jobs

def apply_no_idle_constraint(sequence, processing_times):
    """Apply no-idle constraint to the schedule"""
    processing_times = np.array(processing_times)
    num_jobs = len(sequence)
    num_machines = processing_times.shape[1]
    
    start_times = np.zeros((num_jobs, num_machines))
    completion_times = np.zeros((num_jobs, num_machines))
    
    # First job
    for m in range(num_machines):
        if m == 0:
            start_times[0][m] = 0
        else:
            start_times[0][m] = completion_times[0][m-1]
        completion_times[0][m] = start_times[0][m] + processing_times[0][m]
    
    # Remaining jobs
    for j in range(1, num_jobs):
        for m in range(num_machines):
            if m == 0:
                start_times[j][m] = completion_times[j-1][m]
            else:
                # Ensure no idle time on machine m
                start_times[j][m] = max(completion_times[j][m-1], completion_times[j-1][m])
            completion_times[j][m] = start_times[j][m] + processing_times[j][m]
    
    return start_times, completion_times

def apply_no_wait_constraint(sequence, processing_times, setup_times=None):
    """
    Adjusts the job schedule to ensure no waiting time between operations of the same job in a flowshop environment.

    In a no-wait flowshop, once a job starts, it must be processed continuously without any waiting time between machines.
    This function modifies the schedule to adhere to this constraint by calculating appropriate start and completion times
    for each job on each machine.

    Parameters:
    - sequence (list): A list of job indices representing the order in which jobs should be processed.
    - processing_times (list or np.array): A 2D array where each row corresponds to a job and each column corresponds to
      a machine. The value at [i][j] represents the processing time of job i on machine j.
    - setup_times (list or np.array, optional): A 2D array representing setup times between jobs on machines. This is
      currently not used in the function but can be included for future enhancements.

    Returns:
    - start_times (np.array): A 2D array where each element [i][j] represents the start time of job i on machine j.
    - completion_times (np.array): A 2D array where each element [i][j] represents the completion time of job i on machine j.

    The function assumes that processing_times is a valid 2D array with dimensions corresponding to the number of jobs
    and machines. The sequence list should contain valid job indices.

    Example:
    >>> sequence = [0, 1, 2]
    >>> processing_times = [[2, 3, 2], [4, 1, 3], [3, 2, 1]]
    >>> start_times, completion_times = apply_no_wait_constraint(sequence, processing_times)
    >>> print(start_times)
    >>> print(completion_times)
    """
    processing_times = np.array(processing_times)
    num_jobs = len(sequence)
    num_machines = processing_times.shape[1]
    start_times = np.zeros((num_jobs, num_machines))
    completion_times = np.zeros((num_jobs, num_machines))
    
    # Premier job
    start_time = 0
    for m in range(num_machines):
        start_times[0][m] = start_time
        completion_times[0][m] = start_time + processing_times[sequence[0]][m]
        start_time = completion_times[0][m]
    
    # Jobs suivants
    for j in range(1, num_jobs):
        # Trouver le temps de début minimum qui respecte la contrainte de non-chevauchement
        min_start = 0
        for m in range(num_machines):
            # Calculer le temps de début nécessaire pour cette machine
            prev_completion = completion_times[j-1][m]
            time_before = sum(processing_times[sequence[j]][:m])
            required_start = prev_completion - time_before
            min_start = max(min_start, required_start)
        
        # Planifier toutes les opérations du job en séquence continue
        start_time = min_start
        for m in range(num_machines):
            start_times[j][m] = start_time
            completion_times[j][m] = start_time + processing_times[sequence[j]][m]
            start_time = completion_times[j][m]
    
    return start_times, completion_times

def apply_blocking_constraint(sequence, processing_times, setup_times=None):
    """Adjust schedule to handle blocking between machines"""
    processing_times = np.array(processing_times)
    num_jobs = len(sequence)
    num_machines = processing_times.shape[1]
    start_times = np.zeros((num_jobs, num_machines))
    completion_times = np.zeros((num_jobs, num_machines))
    blocking_times = np.zeros((num_jobs, num_machines))
    
    # First job
    for m in range(num_machines):
        if m == 0:
            start_times[0][m] = 0
        else:
            start_times[0][m] = completion_times[0][m-1]
        completion_times[0][m] = start_times[0][m] + processing_times[0][m]
    
    # Remaining jobs
    for j in range(1, num_jobs):
        for m in range(num_machines):
            if m == 0:
                # First machine
                if setup_times is not None:
                    setup_time = setup_times[sequence[j-1]][sequence[j]][m]
                    start_times[j][m] = completion_times[j-1][m] + setup_time
                else:
                    start_times[j][m] = completion_times[j-1][m]
            else:
                # Check if next machine is available
                if m < num_machines - 1:
                    # Job can't start on current machine until previous job has moved to next machine
                    start_times[j][m] = max(completion_times[j][m-1], 
                                          completion_times[j-1][m],
                                          blocking_times[j-1][m-1])
                else:
                    # Last machine
                    start_times[j][m] = max(completion_times[j][m-1], 
                                          completion_times[j-1][m])
            
            completion_times[j][m] = start_times[j][m] + processing_times[j][m]
            
            # Calculate blocking time if not last machine
            if m < num_machines - 1:
                blocking_times[j][m] = start_times[j][m+1]
    
    return start_times, completion_times

@app.route('/schedule', methods=['POST'])
def schedule():
    data = request.json
    processing_times = np.array(data['processing_times'])
    rule = data['rule']
    constraints = data.get('constraints', {})
    
    # Get additional parameters if provided
    setup_times = np.array(data.get('setup_times')) if 'setup_times' in data else None
    release_times = np.array(data.get('release_times')) if 'release_times' in data else None
    due_dates = np.array(data.get('due_dates')) if 'due_dates' in data else None
    
    # Apply selected scheduling rule
    if rule == 'spt':
        sequence = apply_spt_rule(processing_times)
    elif rule == 'lpt':
        sequence = apply_lpt_rule(processing_times)
    elif rule == 'fifo':
        sequence = FIFO(release_times)
    elif rule == 'lifo':
        sequence = LIFO(release_times)
    elif rule == 'edd':
        sequence = apply_edd_rule(due_dates)
    elif rule == 'cds':
        sequence = apply_cds_rule(processing_times)
    else:
        return jsonify({'error': 'Invalid scheduling rule'})
    
    # Convert sequence to 0-indexed for internal processing
    sequence = [j-1 for j in sequence]
    
    # Apply constraints
    if constraints.get('noIdle'):
        start_times, completion_times = apply_no_idle_constraint(sequence, processing_times, setup_times)
    elif constraints.get('noWait'):
        start_times, completion_times = apply_no_wait_constraint(sequence, processing_times, setup_times)
    elif constraints.get('blocking'):
        start_times, completion_times = apply_blocking_constraint(sequence, processing_times, setup_times)
    else:
        # Calculate regular schedule without constraints
        start_times, completion_times = calculate_schedule(sequence, processing_times, setup_times)
    
    # Calculate metrics
    makespan = float(completion_times[-1][-1])  # Convert to Python float
    machine_utilization = [float(x) for x in calculate_machine_utilization(start_times, completion_times, makespan)]  # Convert to Python float list
    idle_times = [float(x) for x in calculate_idle_times(start_times, completion_times, makespan)]  # Convert to Python float list
    
    # Convert sequence back to 1-indexed for output
    sequence = [int(j+1) for j in sequence]  # Convert to Python int
    
    # Convert numpy arrays to Python lists
    start_times = start_times.tolist()
    completion_times = completion_times.tolist()
    
    return jsonify({
        'sequence': sequence,
        'makespan': makespan,
        'machine_utilization': machine_utilization,
        'idle_times': idle_times,
        'start_times': start_times,
        'completion_times': completion_times
    })

def calculate_schedule(sequence, processing_times, setup_times=None):
    """Calculate regular schedule without special constraints"""
    num_jobs = len(sequence)
    num_machines = processing_times.shape[1]
    start_times = np.zeros((num_jobs, num_machines))
    completion_times = np.zeros((num_jobs, num_machines))
    
    # First job
    for m in range(num_machines):
        if m == 0:
            start_times[0][m] = 0
        else:
            start_times[0][m] = completion_times[0][m-1]
        completion_times[0][m] = start_times[0][m] + processing_times[sequence[0]][m]
    
    # Remaining jobs
    for j in range(1, num_jobs):
        for m in range(num_machines):
            if m == 0:
                if setup_times is not None:
                    setup_time = setup_times[sequence[j-1]][sequence[j]][m]
                    start_times[j][m] = completion_times[j-1][m] + setup_time
                else:
                    start_times[j][m] = completion_times[j-1][m]
            else:
                if setup_times is not None:
                    setup_time = setup_times[sequence[j-1]][sequence[j]][m]
                    start_times[j][m] = max(completion_times[j][m-1],
                                          completion_times[j-1][m] + setup_time)
                else:
                    start_times[j][m] = max(completion_times[j][m-1],
                                          completion_times[j-1][m])
            
            completion_times[j][m] = start_times[j][m] + processing_times[sequence[j]][m]
    
    return start_times, completion_times

def calculate_machine_utilization(start_times, completion_times, makespan):
    """Calculate utilization percentage for each machine"""
    start_times = np.array(start_times)
    completion_times = np.array(completion_times)
    num_machines = start_times.shape[1]
    utilization = np.zeros(num_machines)
    
    for m in range(num_machines):
        total_processing_time = sum(completion_times[:, m] - start_times[:, m])
        utilization[m] = (total_processing_time / makespan) * 100
        
    return utilization

def calculate_idle_times(start_times, completion_times, makespan):
    """Calculate idle time for each machine"""
    start_times = np.array(start_times)
    completion_times = np.array(completion_times)
    num_machines = start_times.shape[1]
    idle_times = np.zeros(num_machines)
    
    for m in range(num_machines):
        busy_time = sum(completion_times[:, m] - start_times[:, m])
        idle_times[m] = makespan - busy_time
        
    return idle_times

class JobShopProblem:
    def __init__(self, n_jobs, n_machines):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.operations = []

    def add_job(self, operations):
        self.operations.append(operations)

def solve_milp_jobshop(problem):
    """
    Solve the job shop scheduling problem using MILP
    """
    try:
        # Create the model
        model = LpProblem("JobShop", LpMinimize)
        
        # Calculate the maximum possible completion time (sum of all processing times)
        max_time = sum(sum(op[1] for op in job) for job in problem.operations)
        
        # Decision Variables
        start_times = {}  # Start time of each operation
        completion_times = {}  # Completion time of each operation
        y = {}  # Binary variable for precedence between operations on the same machine
        
        # Create variables for each operation
        for i in range(problem.n_jobs):
            for j in range(len(problem.operations[i])):
                start_times[f'start_{i}_{j}'] = LpVariable(f'start_{i}_{j}', 0, max_time, LpInteger)
                completion_times[f'completion_{i}_{j}'] = LpVariable(f'completion_{i}_{j}', 0, max_time, LpInteger)

        # Create binary variables for operation pairs on the same machine
        for i1 in range(problem.n_jobs):
            for j1 in range(len(problem.operations[i1])):
                for i2 in range(i1 + 1, problem.n_jobs):
                    for j2 in range(len(problem.operations[i2])):
                        if problem.operations[i1][j1][0] == problem.operations[i2][j2][0]:  # Same machine
                            y[f'y_{i1}_{j1}_{i2}_{j2}'] = LpVariable(f'y_{i1}_{j1}_{i2}_{j2}', 0, 1, LpBinary)

        # Objective: Minimize makespan (maximum completion time)
        makespan = LpVariable('makespan', 0, max_time, LpInteger)
        model += makespan

        # Constraints
        for i in range(problem.n_jobs):
            for j in range(len(problem.operations[i])):
                # Completion time = start time + processing time
                model += completion_times[f'completion_{i}_{j}'] == start_times[f'start_{i}_{j}'] + problem.operations[i][j][1]
                
                # Operation sequence within the same job
                if j > 0:
                    model += start_times[f'start_{i}_{j}'] >= completion_times[f'completion_{i}_{j-1}']
                
                # Makespan constraint
                model += makespan >= completion_times[f'completion_{i}_{j}']

        # Non-overlap constraints for operations on the same machine
        M = max_time  # Big M constant
        for i1 in range(problem.n_jobs):
            for j1 in range(len(problem.operations[i1])):
                for i2 in range(i1 + 1, problem.n_jobs):
                    for j2 in range(len(problem.operations[i2])):
                        if problem.operations[i1][j1][0] == problem.operations[i2][j2][0]:  # Same machine
                            # Either operation (i1,j1) precedes (i2,j2) or vice versa
                            model += start_times[f'start_{i2}_{j2}'] >= completion_times[f'completion_{i1}_{j1}'] - \
                                   M * (1 - y[f'y_{i1}_{j1}_{i2}_{j2}'])
                            model += start_times[f'start_{i1}_{j1}'] >= completion_times[f'completion_{i2}_{j2}'] - \
                                   M * y[f'y_{i1}_{j1}_{i2}_{j2}']

        # Solve the model with CBC solver
        solver = pulp.PULP_CBC_CMD(msg=0)  # Disable solver output
        status = model.solve(solver)

        if status != 1:  # Not optimal
            if status == 0:  # Not solved
                return None, "Problem could not be solved"
            elif status == -1:  # Infeasible
                return None, "Problem is infeasible"
            elif status == -2:  # Unbounded
                return None, "Problem is unbounded"
            else:
                return None, f"Solver returned status: {status}"

        # Extract solution
        solution = {}
        for var in model.variables():
            solution[var.name] = value(var)

        return solution, "optimal"

    except Exception as e:
        print(f"Error in solve_milp_jobshop: {str(e)}")
        return None, f"Error in solve_milp_jobshop: {str(e)}"

def solve_jackson_jobshop(problem):
    """
    Solves the job shop problem using Jackson's Rule with machine availability tracking
    """
    try:
        n_jobs = problem.n_jobs
        n_machines = problem.n_machines
        
        # Calculate total processing times and earliest completion times for each job
        job_metrics = []
        for i, job_ops in enumerate(problem.operations):
            total_time = sum(op[1] for op in job_ops)
            earliest_completion = sum(op[1] for op in job_ops[:1])  # First operation completion
            job_metrics.append((i, total_time, earliest_completion))
        
        # Sort jobs by earliest completion time and total processing time
        job_order = sorted(job_metrics, key=lambda x: (x[2], x[1]))
        job_order = [x[0] for x in job_order]  # Extract job indices
        
        # Initialize machine and job tracking
        machine_completion_times = [0] * n_machines
        job_operation_completion = [0] * n_jobs
        start_times = {}
        completion_times = {}
        
        # Schedule each operation
        for job_idx in job_order:
            for op_idx, operation in enumerate(problem.operations[job_idx]):
                machine = operation[0]
                proc_time = operation[1]
                
                # Start time is the maximum of:
                # 1. Previous operation completion time for this job
                # 2. Machine availability
                earliest_start = job_operation_completion[job_idx]
                machine_available = machine_completion_times[machine]
                start_time = max(earliest_start, machine_available)
                
                # Update times
                completion_time = start_time + proc_time
                start_times[f'start_{job_idx}_{op_idx}'] = start_time
                completion_times[f'completion_{job_idx}_{op_idx}'] = completion_time
                
                # Update tracking variables
                job_operation_completion[job_idx] = completion_time
                machine_completion_times[machine] = completion_time
        
        return {**start_times, **completion_times}
        
    except Exception as e:
        print(f"Error in solve_jackson_jobshop: {str(e)}")
        return None

def solve_jobshop_fifo(problem):
    """
    Solves the job shop problem using FIFO rule (based on job indices)
    """
    try:
        job_order = list(range(problem.n_jobs))
        return schedule_jobshop_sequence(problem, job_order)
    except Exception as e:
        print(f"Error in solve_jobshop_fifo: {str(e)}")
        return None

def solve_jobshop_lifo(problem):
    """
    Solves the job shop problem using LIFO rule (reverse of job indices)
    """
    try:
        job_order = list(range(problem.n_jobs - 1, -1, -1))
        return schedule_jobshop_sequence(problem, job_order)
    except Exception as e:
        print(f"Error in solve_jobshop_lifo: {str(e)}")
        return None

def solve_jobshop_spt(problem):
    """
    Solves the job shop problem using SPT rule
    """
    try:
        # Calculate total processing time for each job
        processing_times = []
        for i, job_ops in enumerate(problem.operations):
            total_time = sum(op[1] for op in job_ops)
            processing_times.append((i, total_time))
        
        # Sort jobs by total processing time
        job_order = [x[0] for x in sorted(processing_times, key=lambda x: x[1])]
        return schedule_jobshop_sequence(problem, job_order)
    except Exception as e:
        print(f"Error in solve_jobshop_spt: {str(e)}")
        return None

def solve_jobshop_lpt(problem):
    """
    Solves the job shop problem using LPT rule
    """
    try:
        # Calculate total processing time for each job
        processing_times = []
        for i, job_ops in enumerate(problem.operations):
            total_time = sum(op[1] for op in job_ops)
            processing_times.append((i, total_time))
        
        # Sort jobs by total processing time (descending)
        job_order = [x[0] for x in sorted(processing_times, key=lambda x: x[1], reverse=True)]
        return schedule_jobshop_sequence(problem, job_order)
    except Exception as e:
        print(f"Error in solve_jobshop_lpt: {str(e)}")
        return None

def solve_jobshop_edd(problem, due_dates):
    """
    Solves the job shop problem using EDD rule
    """
    try:
        # Sort jobs by due dates
        job_order = [x[0] for x in sorted(enumerate(due_dates), key=lambda x: x[1])]
        return schedule_jobshop_sequence(problem, job_order)
    except Exception as e:
        print(f"Error in solve_jobshop_edd: {str(e)}")
        return None

def schedule_jobshop_sequence(problem, job_order):
    """
    Helper function to schedule jobs according to a given sequence
    """
    try:
        n_machines = problem.n_machines
        machine_completion_times = [0] * n_machines
        job_operation_completion = [0] * len(job_order)
        start_times = {}
        completion_times = {}
        
        for job_idx in job_order:
            for op_idx, operation in enumerate(problem.operations[job_idx]):
                machine = operation[0]
                proc_time = operation[1]
                
                # Start time is the maximum of:
                # 1. Previous operation completion time for this job
                # 2. Machine availability
                earliest_start = job_operation_completion[job_idx]
                machine_available = machine_completion_times[machine]
                start_time = max(earliest_start, machine_available)
                
                # Update times
                completion_time = start_time + proc_time
                start_times[f'start_{job_idx}_{op_idx}'] = start_time
                completion_times[f'completion_{job_idx}_{op_idx}'] = completion_time
                
                # Update tracking variables
                job_operation_completion[job_idx] = completion_time
                machine_completion_times[machine] = completion_time
        
        return {**start_times, **completion_times}
        
    except Exception as e:
        print(f"Error in schedule_jobshop_sequence: {str(e)}")
        return None

@app.route('/solve_jobshop', methods=['POST'])
def solve_jobshop():
    try:
        print("Received jobshop request")
        data = request.json
        print("Request data:", data)
        
        if not data or 'n_jobs' not in data or 'n_machines' not in data or 'operations' not in data or 'solve_method' not in data:
            return jsonify({'error': 'Missing required data'}), 400

        # Validate input data
        n_jobs = data['n_jobs']
        n_machines = data['n_machines']
        operations = data['operations']
        solve_method = data['solve_method']
        
        print(f"Processing: {n_jobs} jobs, {n_machines} machines, method: {solve_method}")
        print("Operations:", operations)

        if not (1 <= n_jobs <= 10 and 1 <= n_machines <= 10):
            return jsonify({'error': 'Number of jobs and machines must be between 1 and 10'}), 400

        # Initialize problem
        problem = JobShopProblem(n_jobs=n_jobs, n_machines=n_machines)
        for ops in operations:
            problem.add_job(ops)

        print("Problem initialized")

        # Solve using selected method
        solution = None
        if solve_method == 'milp':
            print("Starting MILP solver")
            solution, status_msg = solve_milp_jobshop(problem)
            print("MILP solution:", solution)
            if not solution:
                return jsonify({'error': f'MILP solver failed: {status_msg}'}), 400
        elif solve_method == 'jackson':
            solution = solve_jackson_jobshop(problem)
        elif solve_method == 'fifo':
            solution = solve_jobshop_fifo(problem)
        elif solve_method == 'lifo':
            solution = solve_jobshop_lifo(problem)
        elif solve_method == 'spt':
            solution = solve_jobshop_spt(problem)
        elif solve_method == 'lpt':
            solution = solve_jobshop_lpt(problem)
        else:
            return jsonify({'error': 'Invalid solve method'}), 400
            
        if not solution:
            return jsonify({'error': 'No feasible solution found'}), 400

        print("Solution found, generating Gantt chart")

        # Generate Gantt chart
        try:
            gantt_img = generate_gantt_chart_jobshop(solution, problem, solve_method)
            print("Gantt chart generated")
        except Exception as e:
            print("Gantt chart generation failed:", str(e))
            return jsonify({'error': f'Failed to generate Gantt chart: {str(e)}'}), 500

        # Calculate metrics
        try:
            print("Calculating metrics")
            makespan = max(solution[f'completion_{i}_{len(problem.operations[i])-1}']
                          for i in range(problem.n_jobs))
            
            flow_times = [solution[f'completion_{i}_{len(problem.operations[i])-1}']
                         for i in range(problem.n_jobs)]
            avg_flow_time = sum(flow_times) / len(flow_times)
            
            # Calculate machine utilization
            machine_spans = {m: [] for m in range(problem.n_machines)}
            for i in range(problem.n_jobs):
                for j, (machine, _) in enumerate(problem.operations[i]):
                    start = solution[f'start_{i}_{j}']
                    end = solution[f'completion_{i}_{j}']
                    machine_spans[machine].append((start, end))
            
            machine_utilization = {}
            for machine, spans in machine_spans.items():
                total_busy_time = sum(end - start for start, end in spans)
                machine_utilization[machine] = (total_busy_time / makespan) * 100

            metrics = {
                'makespan': makespan,
                'avg_flow_time': avg_flow_time,
                'machine_utilization': machine_utilization
            }
            
            print("Metrics calculated:", metrics)

            response_data = {
                'status': 'success',
                'metrics': metrics,
                'gantt_chart': gantt_img
            }
            print("Sending response")
            return jsonify(response_data)

        except Exception as e:
            print("Metrics calculation failed:", str(e))
            return jsonify({'error': f'Failed to calculate metrics: {str(e)}'}), 500

    except Exception as e:
        print("General error:", str(e))
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_gantt_chart_jobshop(solution, problem, solve_method='milp', due_dates=None):
    """
    Generate a Gantt chart for the jobshop solution with enhanced visualization
    
    Args:
        solution: Dictionary containing start and completion times
        problem: JobShopProblem instance
        solve_method: The scheduling rule used (milp, jackson, fifo, etc.)
        due_dates: List of due dates for EDD rule visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from io import BytesIO
    import base64
    import numpy as np

    # Calculate figure dimensions based on number of machines
    height_per_machine = 1
    fig_height = max(6, problem.n_machines * height_per_machine)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, fig_height + 2), 
                                  gridspec_kw={'height_ratios': [fig_height, 2]})
    
    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, problem.n_jobs))
    
    # Track machine utilization
    machine_spans = {m: [] for m in range(problem.n_machines)}
    
    # Calculate makespan
    makespan = max(solution[f'completion_{i}_{len(problem.operations[i])-1}']
                  for i in range(problem.n_jobs))

    # Plot operations
    for job_id in range(problem.n_jobs):
        for op_id, (machine, proc_time) in enumerate(problem.operations[job_id]):
            start_time = solution[f'start_{job_id}_{op_id}']
            end_time = solution[f'completion_{job_id}_{op_id}']
            
            # Main operation block
            rect = patches.Rectangle(
                (start_time, machine - 0.4),
                proc_time,
                0.8,
                facecolor=colors[job_id],
                edgecolor='black',
                label=f'Job {job_id + 1}' if op_id == 0 else None,
                alpha=0.8
            )
            ax1.add_patch(rect)
            
            # Add job and operation information
            ax1.text(start_time + proc_time/2, machine,
                    f'J{job_id + 1}\nOp{op_id + 1}\n{proc_time}',
                    ha='center', va='center',
                    fontsize=8)
            
            # Track machine utilization
            machine_spans[machine].append((start_time, end_time))
    
    # Add timeline visualization in the bottom subplot
    for job_id in range(problem.n_jobs):
        first_start = solution[f'start_{job_id}_0']
        last_completion = solution[f'completion_{job_id}_{len(problem.operations[job_id])-1}']
        
        # Plot job timeline
        ax2.broken_barh([(first_start, last_completion - first_start)],
                       (job_id - 0.4, 0.8),
                       facecolors=colors[job_id],
                       alpha=0.8)
        
        # Add completion time
        ax2.text(last_completion, job_id,
                f'C{job_id + 1}={last_completion:.1f}',
                va='center', ha='left',
                fontsize=8)
        
        # Add due date marker for EDD
        if solve_method == 'edd' and due_dates is not None:
            ax2.axvline(x=due_dates[job_id], 
                       ymin=(job_id - 0.4)/problem.n_jobs,
                       ymax=(job_id + 0.4)/problem.n_jobs,
                       color='red', linestyle='--', alpha=0.5)
            ax2.text(due_dates[job_id], job_id,
                    f'd{job_id + 1}={due_dates[job_id]}',
                    va='bottom', ha='right',
                    fontsize=8, color='red')

    # Calculate and display machine utilization
    for machine, spans in machine_spans.items():
        total_busy_time = sum(end - start for start, end in spans)
        utilization = (total_busy_time / makespan) * 100
        ax1.text(makespan * 1.02, machine,
                f'Util: {utilization:.1f}%',
                va='center', ha='left',
                fontsize=8)

    # Customize the main Gantt chart
    ax1.set_ylim(-1, problem.n_machines + 0.5)
    ax1.set_xlim(0, makespan * 1.15)
    ax1.set_ylabel('Machine')
    ax1.set_xlabel('Time')
    ax1.set_title(f'Job Shop Schedule - {solve_method.upper()} Rule\nMakespan: {makespan:.1f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Customize the timeline subplot
    ax2.set_ylim(-1, problem.n_jobs + 0.5)
    ax2.set_xlim(0, makespan * 1.15)
    ax2.set_ylabel('Job')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    # Add rule-specific information
    info_text = f"Schedule Information:\n"
    info_text += f"- Method: {solve_method.upper()}\n"
    info_text += f"- Makespan: {makespan:.1f}\n"
    
    # Add specific metrics based on the rule
    if solve_method == 'edd' and due_dates is not None:
        tardiness = [max(0, solution[f'completion_{i}_{len(problem.operations[i])-1}'] - due_dates[i])
                    for i in range(problem.n_jobs)]
        max_tardiness = max(tardiness)
        total_tardiness = sum(tardiness)
        info_text += f"- Max Tardiness: {max_tardiness:.1f}\n"
        info_text += f"- Total Tardiness: {total_tardiness:.1f}"
    elif solve_method in ['spt', 'lpt']:
        total_processing_times = [sum(op[1] for op in problem.operations[i])
                                for i in range(problem.n_jobs)]
        info_text += f"- Min Proc. Time: {min(total_processing_times):.1f}\n"
        info_text += f"- Max Proc. Time: {max(total_processing_times):.1f}"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.8))

    # Adjust layout and convert to image
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return base64.b64encode(image_png).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
