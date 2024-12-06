// Flow Shop Scheduling JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize event listeners
    document.getElementById('generate_tables').addEventListener('click', generateTables);
    document.getElementById('scheduler-form').addEventListener('submit', handleScheduling);
    document.getElementById('export_results').addEventListener('click', exportResults);
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

async function handleScheduling(e) {
    e.preventDefault();
    
    const numJobs = parseInt(document.getElementById('num_jobs').value);
    const numMachines = parseInt(document.getElementById('num_machines').value);
    const rule = document.getElementById('scheduling_rule').value;
    
    // Collect processing times
    const processingTimes = Array.from({length: numJobs}, () => 
        Array.from({length: numMachines}, () => 0)
    );
    
    document.querySelectorAll('.processing-time').forEach(input => {
        const job = parseInt(input.dataset.job);
        const machine = parseInt(input.dataset.machine);
        processingTimes[job][machine] = parseInt(input.value) || 0;
    });
    
    // Collect additional data based on selected rule
    let additionalData = {};
    
    if (rule === 'fifo' || rule === 'lifo') {
        const releaseTimes = Array(numJobs).fill(0);
        document.querySelectorAll('.release-time').forEach(input => {
            const job = parseInt(input.dataset.job);
            releaseTimes[job] = parseInt(input.value) || 0;
        });
        additionalData.release_times = releaseTimes;
    }
    
    if (rule === 'edd') {
        const dueDates = Array(numJobs).fill(0);
        document.querySelectorAll('.due-date').forEach(input => {
            const job = parseInt(input.dataset.job);
            dueDates[job] = parseInt(input.value) || 0;
        });
        additionalData.due_dates = dueDates;
    }
    
    // Collect setup times if enabled
    if (document.getElementById('setup_times').checked) {
        const setupTimes = Array.from({length: numJobs}, () => 
            Array.from({length: numJobs}, () => 
                Array.from({length: numMachines}, () => 0)
            )
        );
        
        document.querySelectorAll('.setup-time').forEach(input => {
            const from = parseInt(input.dataset.from);
            const to = parseInt(input.dataset.to);
            const machine = parseInt(input.dataset.machine);
            setupTimes[from][to][machine] = parseInt(input.value) || 0;
        });
        
        additionalData.setup_times = setupTimes;
    }
    
    // Collect constraints
    const constraints = {
        noWait: document.getElementById('no_wait').checked,
        noIdle: document.getElementById('no_idle').checked,
        blocking: document.getElementById('blocking').checked
    };
    
    try {
        const response = await fetch('/schedule', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                processing_times: processingTimes,
                rule: rule,
                constraints: constraints,
                ...additionalData
            })
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
        
        // Display makespan
        document.getElementById('makespan').textContent = 
            `Total completion time: ${data.makespan.toFixed(2)} units`;
        
        // Display machine utilization
        document.getElementById('machine_utilization').innerHTML = 
            data.machine_utilization.map((util, i) => 
                `Machine ${i + 1}: ${util.toFixed(2)}% utilization (Idle time: ${data.idle_times[i].toFixed(2)} units)`
            ).join('<br>');
        
        // Hide any previous error message
        document.getElementById('gantt_chart_error').classList.add('hidden');
        
        // Display Gantt chart
        const ganttData = generateGanttData(data);
        Plotly.newPlot('gantt_chart', ganttData.data, ganttData.layout)
            .catch(error => {
                console.error('Error plotting Gantt chart:', error);
                document.getElementById('gantt_chart_error').textContent = 
                    'Error displaying Gantt chart. Please check the console for details.';
                document.getElementById('gantt_chart_error').classList.remove('hidden');
            });
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
    
    try {
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
                zeroline: false
            },
            height: 400,
            margin: {
                l: 100,
                r: 50,
                t: 50,
                b: 50
            },
            hovermode: 'closest'
        };
        
        return { data: plotData, layout };
    } catch (error) {
        console.error('Error generating Gantt data:', error);
        throw error;
    }
}

function exportResults() {
    const resultsSection = document.getElementById('results_section');
    
    // Create a copy of the results section
    const content = resultsSection.cloneNode(true);
    
    // Remove the Plotly chart (it will be replaced with an image)
    const ganttChart = content.querySelector('#gantt_chart');
    if (ganttChart) {
        const img = document.createElement('img');
        img.src = Plotly.toImage('gantt_chart', {format: 'png', width: 800, height: 400})
            .then(url => img.src = url);
        ganttChart.parentNode.replaceChild(img, ganttChart);
    }
    
    // Create and download HTML file
    const html = `
        <!DOCTYPE html>
        <html>
        <head>
            <title>Flow Shop Scheduling Results</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="p-8">
            ${content.outerHTML}
        </body>
        </html>
    `;
    
    const blob = new Blob([html], { type: 'text/html' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scheduling_results.html';
    a.click();
    window.URL.revokeObjectURL(url);
}


