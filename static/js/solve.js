// Flow Shop Scheduling JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize event listeners
    document.getElementById('generate_tables').addEventListener('click', generateTables);
    document.getElementById('scheduler-form').addEventListener('submit', handleScheduling);
    document.getElementById('scheduling_rule').addEventListener('change', handleRuleChange);
    document.getElementById('setup_times').addEventListener('change', handleSetupTimesChange);
});

function handleRuleChange() {
    const rule = document.getElementById('scheduling_rule').value;
    const releaseTimes = document.getElementById('release_times_container');
    const dueDates = document.getElementById('due_dates_container');
    
    // Show/hide relevant input tables based on selected rule
    if (rule === 'fifo' || rule === 'lifo') {
        releaseTimes.classList.remove('hidden');
        dueDates.classList.add('hidden');
    } else if (rule === 'edd') {
        dueDates.classList.remove('hidden');
        releaseTimes.classList.add('hidden');
    } else {
        releaseTimes.classList.add('hidden');
        dueDates.classList.add('hidden');
    }
}

function handleSetupTimesChange() {
    const setupTimesChecked = document.getElementById('setup_times').checked;
    const setupTimesContainer = document.getElementById('setup_times_container');
    
    if (setupTimesChecked) {
        setupTimesContainer.classList.remove('hidden');
        generateSetupTimesTable();
    } else {
        setupTimesContainer.classList.add('hidden');
    }
}

function generateTables() {
    const numJobs = parseInt(document.getElementById('num_jobs').value);
    const numMachines = parseInt(document.getElementById('num_machines').value);
    
    if (!numJobs || !numMachines) {
        alert('Please enter valid numbers for jobs and machines');
        return;
    }
    
    // Generate Processing Times Table
    generateProcessingTimesTable(numJobs, numMachines);
    
    // Generate Release Times Table if needed
    const rule = document.getElementById('scheduling_rule').value;
    if (rule === 'fifo' || rule === 'lifo') {
        generateReleaseTimesTable(numJobs);
    }
    
    // Generate Due Dates Table if needed
    if (rule === 'edd') {
        generateDueDatesTable(numJobs);
    }
    
    // Generate Setup Times Table if needed
    if (document.getElementById('setup_times').checked) {
        generateSetupTimesTable(numJobs, numMachines);
    }
}

function generateProcessingTimesTable(numJobs, numMachines) {
    const table = document.getElementById('processing_times_table');
    const container = document.getElementById('processing_times_container');
    
    let html = `
        <thead>
            <tr>
                <th class="px-4 py-2">Job</th>
                ${Array.from({length: numMachines}, (_, i) => 
                    `<th class="px-4 py-2">Machine ${i + 1}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
    `;
    
    for (let i = 0; i < numJobs; i++) {
        html += `
            <tr>
                <td class="px-4 py-2">Job ${i + 1}</td>
                ${Array.from({length: numMachines}, (_, j) => `
                    <td class="px-4 py-2">
                        <input type="number" 
                               class="processing-time w-20 px-2 py-1 border border-gray-300 rounded"
                               data-job="${i}"
                               data-machine="${j}"
                               min="0"
                               required>
                    </td>
                `).join('')}
            </tr>
        `;
    }
    
    html += '</tbody>';
    table.innerHTML = html;
    container.classList.remove('hidden');
}

function generateReleaseTimesTable(numJobs) {
    const table = document.getElementById('release_times_table');
    const container = document.getElementById('release_times_container');
    
    let html = `
        <thead>
            <tr>
                <th class="px-4 py-2">Job</th>
                <th class="px-4 py-2">Release Time</th>
            </tr>
        </thead>
        <tbody>
    `;
    
    for (let i = 0; i < numJobs; i++) {
        html += `
            <tr>
                <td class="px-4 py-2">Job ${i + 1}</td>
                <td class="px-4 py-2">
                    <input type="number" 
                           class="release-time w-20 px-2 py-1 border border-gray-300 rounded"
                           data-job="${i}"
                           min="0"
                           required>
                </td>
            </tr>
        `;
    }
    
    html += '</tbody>';
    table.innerHTML = html;
    container.classList.remove('hidden');
}

function generateDueDatesTable(numJobs) {
    const table = document.getElementById('due_dates_table');
    const container = document.getElementById('due_dates_container');
    
    let html = `
        <thead>
            <tr>
                <th class="px-4 py-2">Job</th>
                <th class="px-4 py-2">Due Date</th>
            </tr>
        </thead>
        <tbody>
    `;
    
    for (let i = 0; i < numJobs; i++) {
        html += `
            <tr>
                <td class="px-4 py-2">Job ${i + 1}</td>
                <td class="px-4 py-2">
                    <input type="number" 
                           class="due-date w-20 px-2 py-1 border border-gray-300 rounded"
                           data-job="${i}"
                           min="0"
                           required>
                </td>
            </tr>
        `;
    }
    
    html += '</tbody>';
    table.innerHTML = html;
    container.classList.remove('hidden');
}

function generateSetupTimesTable(numJobs, numMachines) {
    const table = document.getElementById('setup_times_table');
    const container = document.getElementById('setup_times_container');
    
    let html = '';
    
    // Generate table for each machine
    for (let m = 0; m < numMachines; m++) {
        html += `
            <div class="mb-6">
                <h4 class="font-medium mb-2">Machine ${m + 1}</h4>
                <table class="min-w-full divide-y divide-gray-200 mb-4">
                    <thead>
                        <tr>
                            <th class="px-4 py-2">From/To</th>
                            ${Array.from({length: numJobs}, (_, i) => 
                                `<th class="px-4 py-2">Job ${i + 1}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        for (let i = 0; i < numJobs; i++) {
            html += `
                <tr>
                    <td class="px-4 py-2">Job ${i + 1}</td>
                    ${Array.from({length: numJobs}, (_, j) => `
                        <td class="px-4 py-2">
                            <input type="number" 
                                   class="setup-time w-16 px-2 py-1 border border-gray-300 rounded"
                                   data-from="${i}"
                                   data-to="${j}"
                                   data-machine="${m}"
                                   min="0"
                                   ${i === j ? 'disabled value="0"' : 'required'}>
                        </td>
                    `).join('')}
                </tr>
            `;
        }
        
        html += '</tbody></table></div>';
    }
    
    table.innerHTML = html;
    container.classList.remove('hidden');
}

function getProcessingTimes() {
    const numJobs = parseInt(document.getElementById('num_jobs').value);
    const numMachines = parseInt(document.getElementById('num_machines').value);
    const processingTimes = [];
    
    for (let i = 0; i < numJobs; i++) {
        const jobTimes = [];
        for (let j = 0; j < numMachines; j++) {
            const input = document.querySelector(`input[data-job="${i}"][data-machine="${j}"]`);
            if (!input) {
                throw new Error(`Missing processing time input for job ${i + 1}, machine ${j + 1}`);
            }
            const time = parseInt(input.value);
            if (isNaN(time) || time < 0) {
                throw new Error(`Invalid processing time for job ${i + 1}, machine ${j + 1}`);
            }
            jobTimes.push(time);
        }
        processingTimes.push(jobTimes);
    }
    return processingTimes;
}

async function handleScheduling(e) {
    e.preventDefault();
    
    try {
        // Collect form data
        const formData = {
            rule: document.getElementById('scheduling_rule').value,
            num_jobs: parseInt(document.getElementById('num_jobs').value),
            num_machines: parseInt(document.getElementById('num_machines').value),
            no_wait: document.getElementById('no_wait').checked,
            no_idle: document.getElementById('no_idle').checked,
            blocking: document.getElementById('blocking').checked
        };

        // Get processing times
        try {
            formData.processing_times = getProcessingTimes();
        } catch (error) {
            alert(error.message);
            return;
        }

        // Send request to server
        const response = await fetch('/solve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while scheduling jobs');
    }
}

function displayResults(data) {
    try {
        // Show results section
        document.getElementById('results_section').classList.remove('hidden');
        
        // Display job sequence
        document.getElementById('job_sequence').textContent = 
            data.sequence.map(j => `Job ${j}`).join(' → ');
        
        // Display performance metrics
        document.getElementById('makespan').textContent = 
            `${data.makespan.toFixed(2)} units`;
            
        document.getElementById('total_flow_time').textContent = 
            `${data.total_flow_time.toFixed(2)} units`;
            
        document.getElementById('avg_flow_time').textContent = 
            `${data.avg_flow_time.toFixed(2)} units`;
            
        document.getElementById('tfr').textContent = 
            data.tfr.toFixed(3);
            
        document.getElementById('tar').textContent = 
            data.tar.toFixed(3);
        
        // Display machine utilization
        document.getElementById('machine_utilization').innerHTML = 
            data.machine_utilization.map((util, i) => 
                `Machine ${i + 1}: ${util.toFixed(2)}% utilization (Idle time: ${data.idle_times[i].toFixed(2)} units)`
            ).join('<br>');
        
        // Generate and display Gantt chart
        const ganttData = generateGanttData(data);
        Plotly.newPlot('gantt_chart', ganttData.data, ganttData.layout);
        
    } catch (error) {
        console.error('Error displaying results:', error);
        alert('Error displaying results. Please check the console for details.');
    }
}

function generateGanttData(data) {
    const colors = [
        '#2196F3', '#4CAF50', '#FFC107', '#E91E63', '#9C27B0',
        '#00BCD4', '#FF5722', '#795548', '#607D8B', '#3F51B5'
    ];
    
    const plotData = [];
    const numMachines = data.machine_utilization.length;
    
    for (let m = 0; m < numMachines; m++) {
        for (let j = 0; j < data.sequence.length; j++) {
            const jobIndex = data.sequence[j] - 1;
            const start = data.start_times[j][m];
            const end = data.completion_times[j][m];
            const duration = end - start;
            
            plotData.push({
                x: [start, end],
                y: [`Machine ${m + 1}`, `Machine ${m + 1}`],
                mode: 'lines',
                line: {
                    color: colors[jobIndex % colors.length],
                    width: 20
                },
                name: `Job ${data.sequence[j]}`,
                showlegend: m === 0,  // Show legend only once per job
                hovertemplate: 
                    `Job ${data.sequence[j]}<br>` +
                    `Machine ${m + 1}<br>` +
                    `Start: %{x[0]:.2f}<br>` +
                    `End: %{x[1]:.2f}<br>` +
                    `Duration: ${duration.toFixed(2)}<br>` +
                    `<extra></extra>`  // Removes secondary box
            });
        }
    }
    
    const layout = {
        title: 'Gantt Chart',
        xaxis: {
            title: 'Time',
            showgrid: true,
            zeroline: true
        },
        yaxis: {
            showgrid: true,
            zeroline: true,
            autorange: 'reversed'  // Reverses the order of machines
        },
        height: 400,
        margin: { t: 50, b: 50, l: 100, r: 50 },
        showlegend: true,
        legend: {
            traceorder: 'normal',
            font: { size: 10 },
            orientation: 'h'
        }
    };
    
    return { data: plotData, layout: layout };
}

function exportToPDF() {
    try {
        // Create a new jsPDF instance
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        // Set title
        doc.setFontSize(18);
        doc.text('Flow Shop Scheduling Results', 105, 20, { align: 'center' });

        // Job Sequence
        doc.setFontSize(12);
        const jobSequence = document.getElementById('job_sequence').innerText;
        doc.text('Job Sequence:', 20, 40);
        doc.text(jobSequence, 20, 50);

        // Performance Metrics
        const metrics = [
            { label: 'Makespan (Cmax)', id: 'makespan' },
            { label: 'Total Flow Time', id: 'total_flow_time' },
            { label: 'Average Flow Time', id: 'avg_flow_time' },
            { label: 'Total Flow Ratio (TFR)', id: 'tfr' },
            { label: 'Time Average Ratio (TAR)', id: 'tar' }
        ];

        doc.text('Performance Metrics:', 20, 70);
        let y = 80;
        metrics.forEach(metric => {
            const value = document.getElementById(metric.id).innerText;
            doc.text(`${metric.label}: ${value}`, 25, y);
            y += 10;
        });

        // Machine Utilization
        doc.text('Machine Utilization:', 20, y + 10);
        const utilization = document.getElementById('machine_utilization').innerText;
        y += 20;
        doc.text(utilization, 25, y);

        // Save PDF
        doc.save('flowshop_schedule.pdf');
    } catch (error) {
        console.error('PDF Export Error:', error);
        alert('Failed to export PDF. Please try again.');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const exportButton = document.getElementById('export_results');
    if (exportButton) {
        exportButton.addEventListener('click', exportToPDF);
    }
});
