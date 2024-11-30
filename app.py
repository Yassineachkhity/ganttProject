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
        
        makespan = completion_times[-1][-1]
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_sequence = sequence
    
    return best_sequence if best_sequence is not None else list(range(num_jobs))


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
                job_order = np.argsort(-np.sum(processing_times, axis=1))  # Order by longest processing time
            elif rule == "edd":
                # For EDD, we'll use total processing time as due date estimation
                due_dates = np.sum(processing_times, axis=1)
                job_order = np.argsort(due_dates)  # Order by earliest due date
            elif rule == "johnson":
                if num_machines == 2:
                    machine1_times = processing_times[:, 0]
                    machine2_times = processing_times[:, 1]
                    set1 = [(i, t) for i, t in enumerate(machine1_times) if t <= machine2_times[i]]
                    set2 = [(i, t) for i, t in enumerate(machine1_times) if t > machine2_times[i]]
                    
                    set1.sort(key=lambda x: x[1])  # Sort set1 by machine1 times
                    set2.sort(key=lambda x: -machine2_times[x[0]])  # Sort set2 by machine2 times in descending order
                    
                    job_order = [job[0] for job in set1 + set2]
                else:
                    return jsonify({"error": "Johnson's rule works only for 2 machines"}), 400
            elif rule == "cds":
                job_order = apply_cds_rule(processing_times)
            else:
                return jsonify({"error": "Invalid scheduling rule selected"}), 400

            # Apply the job order to processing times
            ordered_processing_times = processing_times[job_order]

            # Calculate completion times for each job on each machine with constraints
            completion_times = np.zeros_like(ordered_processing_times)
            start_times = np.zeros_like(ordered_processing_times)
            machine_available_times = np.zeros(num_machines)  # When each machine becomes available
            job_completion_times = np.zeros(num_jobs)  # When each job completes its current operation

            for i in range(num_jobs):
                for j in range(num_machines):
                    # Calculate earliest possible start time
                    if j == 0:  # First machine
                        start_time = machine_available_times[j]
                    else:
                        # Consider previous operation of the same job
                        start_time = max(job_completion_times[i], machine_available_times[j])
                        
                        if no_wait:
                            # Must start immediately after previous operation
                            start_time = job_completion_times[i]
                        
                        if blocking:
                            # Must wait for next machine to be available
                            if j < num_machines - 1:
                                start_time = max(start_time, machine_available_times[j+1])
                    
                    if no_idle and i > 0:
                        # Machine cannot be idle between jobs
                        start_time = max(start_time, machine_available_times[j])
                    
                    # Calculate completion time
                    completion_time = start_time + ordered_processing_times[i][j]
                    
                    # Update times
                    start_times[i][j] = start_time
                    completion_times[i][j] = completion_time
                    machine_available_times[j] = completion_time
                    job_completion_times[i] = completion_time

            # Calculate metrics
            total_time = np.max(completion_times)  # Makespan
            total_flow_time = np.sum(completion_times[:, -1])  # Sum of completion times
            tfr = total_flow_time / (num_jobs * num_machines)
            tar = total_flow_time / num_jobs
            c_max = total_time

            # Generate Gantt chart with start and completion times
            gantt_chart = generate_gantt_chart(job_order, ordered_processing_times, start_times)

            # Return the results as JSON
            return jsonify({
                "total_time": float(total_time),
                "total_flow_time": float(total_flow_time),
                "tfr": float(tfr),
                "tar": float(tar),
                "c_max": float(c_max),
                "gantt_chart": gantt_chart,
                "job_order": [int(i + 1) for i in job_order]  # Add 1 to make it 1-based for display
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template('solve.html')


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


if __name__ == '__main__':
    app.run(debug=True)
