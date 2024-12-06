from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

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
    """Adjust schedule to ensure no waiting time between operations of the same job"""
    processing_times = np.array(processing_times)
    num_jobs = len(sequence)
    num_machines = processing_times.shape[1]
    start_times = np.zeros((num_jobs, num_machines))
    completion_times = np.zeros((num_jobs, num_machines))
    
    # First job
    current_time = 0
    for m in range(num_machines):
        start_times[0][m] = current_time
        completion_times[0][m] = start_times[0][m] + processing_times[0][m]
        current_time = completion_times[0][m]
    
    # Remaining jobs
    for j in range(1, num_jobs):
        if setup_times is not None:
            setup_time = setup_times[sequence[j-1]][sequence[j]][0]
            current_time = completion_times[j-1][0] + setup_time
        else:
            current_time = completion_times[j-1][0]
            
        # Ensure continuous processing across machines
        for m in range(num_machines):
            start_times[j][m] = current_time
            completion_times[j][m] = start_times[j][m] + processing_times[j][m]
            current_time = completion_times[j][m]
    
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

if __name__ == '__main__':
    app.run(debug=False)
